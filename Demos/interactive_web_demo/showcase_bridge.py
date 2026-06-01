"""
showcase_bridge.py
FMS Showcase — bridges Arduino FSR sensors to the pain_simulator.html web demo.

Architecture:
    SerialReader thread  →  asyncio.Queue  →  WebSocket server (port 8765)
        ↓                                           ↓
    Reads Arduino JSON                   Broadcasts scenario ID to
    {"sensor":N,"pressure":V}            all connected browser clients

Usage:
    python showcase_bridge.py                          # auto-detect COM port
    python showcase_bridge.py --port COM3              # specify port
    python showcase_bridge.py --port COM3 --ws-port 8765

Payload broadcast to browser:
    {"scenario": "leg_medium", "sensor": 1, "pressure_raw": 480,
     "intensity": "medium", "location": "leg"}
"""

import argparse
import asyncio
import json
import threading
import time
import serial
import serial.tools.list_ports
import websockets

# ── Per-sensor calibrated thresholds ─────────────────────────────────────────
# Measured resting: Head=4, Torso=5, RightArm=331, LeftArm=24, RightLeg=19, LeftLeg=4
# Measured max:     Head=1014, Torso=1015, RightArm=997, LeftArm=987, RightLeg=988, LeftLeg=997
#
# FSRs have a non-linear response. Bands are weighted toward wider light/medium zones:
#   light  = noise_floor → ~580  (gentle contact)
#   medium = ~580 → ~800         (deliberate press)
#   hard   = ~800+               (firm press)
#
# Format: (noise_floor, light_max, medium_max)
SENSOR_THRESHOLDS = {
    0: (60,  580, 800),   # Head
    1: (60,  580, 800),   # Torso
    2: (390, 650, 830),   # Right Arm (higher resting — narrower light band)
    3: (80,  580, 800),   # Left Arm
    4: (70,  580, 800),   # Right Leg
    5: (60,  580, 800),   # Left Leg
}

# ── Sensor → body location mapping ───────────────────────────────────────────
# Maps sensor index to the scenario prefix (left/right share the same data)
SENSOR_LOCATIONS = {
    0: 'head',   # A0 — head/neck
    1: 'torso',  # A1 — torso
    2: 'arm',    # A2 — right arm
    3: 'arm',    # A3 — left arm  (same scenarios as right — symmetric)
    4: 'leg',    # A4 — right leg
    5: 'leg',    # A5 — left leg  (same scenarios as right — symmetric)
}

# Human-readable label sent in the broadcast payload
SENSOR_LABELS = {
    0: 'head',
    1: 'torso',
    2: 'right_arm',
    3: 'left_arm',
    4: 'right_leg',
    5: 'left_leg',
}

# ── Serial settings ───────────────────────────────────────────────────────────
BAUD_RATE      = 115200
RECONNECT_DELAY_S = 2


# ── Pressure helpers ──────────────────────────────────────────────────────────

def pressure_to_intensity(raw: int, sensor_id: int = 0):
    """
    Convert a raw ADC reading (0–1023) to a pressure intensity label.

    Returns 'light', 'medium', 'hard', or None (no contact).
    Uses per-sensor calibrated thresholds.
    """
    noise_floor, light_max, medium_max = SENSOR_THRESHOLDS.get(sensor_id, (60, 378, 696))
    if raw < noise_floor:
        return None
    if raw <= light_max:
        return 'light'
    if raw <= medium_max:
        return 'medium'
    return 'hard'


def map_to_scenario(sensor_id: int, raw: int):
    """
    Map a sensor ID + raw ADC reading to a scenario string.

    Returns e.g. 'leg_medium', or None if pressure below noise floor.
    """
    intensity = pressure_to_intensity(raw, sensor_id)
    if intensity is None:
        return None
    location = SENSOR_LOCATIONS.get(sensor_id)
    if location is None:
        return None
    return f'{location}_{intensity}'


# ── Auto-detect Arduino serial port ──────────────────────────────────────────

def auto_detect_port():
    """
    Return the first USB serial port that looks like an Arduino, or None.
    Checks description and hardware ID strings for common Arduino identifiers.
    """
    candidates = []
    for port in serial.tools.list_ports.comports():
        desc = (port.description or '').lower()
        hwid = (port.hwid or '').lower()
        if any(kw in desc or kw in hwid for kw in
               ('arduino', 'ch340', 'cp210', 'ftdi', 'usb serial')):
            candidates.append(port.device)

    if candidates:
        print(f'[bridge] Auto-detected ports: {candidates}')
        return candidates[0]

    # Fallback: return the first available COM port
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if ports:
        print(f'[bridge] No Arduino keyword found; trying first port: {ports[0]}')
        return ports[0]

    return None


# ── Serial reader thread ──────────────────────────────────────────────────────

def serial_reader(port: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Blocking thread: opens serial port, reads JSON lines, puts parsed dicts
    onto the asyncio queue.  Auto-reconnects on disconnect.
    """
    while True:
        try:
            print(f'[serial] Connecting to {port} at {BAUD_RATE} baud…')
            with serial.Serial(port, BAUD_RATE, timeout=1) as ser:
                print(f'[serial] Connected.')
                while True:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Put onto the asyncio event loop's queue thread-safely
                        asyncio.run_coroutine_threadsafe(queue.put(data), loop)
                    except json.JSONDecodeError:
                        pass  # ignore malformed lines

        except serial.SerialException as e:
            print(f'[serial] Disconnected: {e}. Retrying in {RECONNECT_DELAY_S} s…')
            time.sleep(RECONNECT_DELAY_S)


# ── WebSocket server ──────────────────────────────────────────────────────────

# Global set of currently connected WebSocket clients
connected_clients: set = set()


async def ws_handler(websocket):
    """Register and deregister WebSocket clients."""
    connected_clients.add(websocket)
    print(f'[ws] Client connected ({len(connected_clients)} total)')
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f'[ws] Client disconnected ({len(connected_clients)} remaining)')


async def broadcast_loop(queue: asyncio.Queue):
    """
    Consume Arduino messages from the queue, map to scenario IDs, and
    broadcast to all connected WebSocket clients.

    Only broadcasts when the scenario changes (debounce), to avoid flooding
    the browser with repeated identical messages.
    """
    last_scenario      = None
    pressure_tick      = 0   # counter for continuous pressure updates

    while True:
        # Wait up to 500 ms for a reading — if nothing arrives the sensor was released
        try:
            data = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if last_scenario is not None:
                print('[bridge] Sensor released (timeout)')
                release_payload = json.dumps({'released': True})
                dead = set()
                for ws in connected_clients.copy():
                    try:
                        await ws.send(release_payload)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(ws)
                connected_clients.difference_update(dead)
                last_scenario = None
                pressure_tick = 0
            continue

        sensor_id = data.get('sensor')
        raw       = data.get('pressure', 0)

        scenario  = map_to_scenario(sensor_id, raw)
        intensity = pressure_to_intensity(raw, sensor_id)

        if scenario is None:
            # Sensor released
            if last_scenario is not None:
                release_payload = json.dumps({'released': True})
                dead = set()
                for ws in connected_clients.copy():
                    try:
                        await ws.send(release_payload)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(ws)
                connected_clients.difference_update(dead)
            last_scenario = None
            pressure_tick = 0
            continue

        # Send continuous pressure updates every 3 reads (~150 ms) so the
        # browser can scale animation speed in real time with live pressure
        pressure_tick += 1
        if pressure_tick >= 3:
            pressure_tick = 0
            p_payload = json.dumps({
                'pressure_update': True,
                'sensor':          sensor_id,
                'pressure_raw':    raw,
                'intensity':       intensity,
            })
            dead = set()
            for ws in connected_clients.copy():
                try:
                    await ws.send(p_payload)
                except websockets.exceptions.ConnectionClosed:
                    dead.add(ws)
            connected_clients.difference_update(dead)

        if scenario == last_scenario:
            continue  # scenario unchanged — pressure update already sent above

        last_scenario = scenario
        location      = SENSOR_LABELS.get(sensor_id, 'unknown')

        payload = json.dumps({
            'scenario':     scenario,
            'sensor':       sensor_id,
            'pressure_raw': raw,
            'intensity':    intensity,
            'location':     location,
        })

        print(f'[bridge] → {payload}')

        # Broadcast to all clients; remove any that have disconnected
        dead = set()
        for ws in connected_clients.copy():
            try:
                await ws.send(payload)
            except websockets.exceptions.ConnectionClosed:
                dead.add(ws)
        connected_clients.difference_update(dead)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(serial_port: str, ws_port: int):
    queue = asyncio.Queue()
    loop  = asyncio.get_running_loop()

    # Start serial reader on a background thread
    t = threading.Thread(
        target=serial_reader,
        args=(serial_port, queue, loop),
        daemon=True,
    )
    t.start()

    print(f'[bridge] WebSocket server starting on ws://localhost:{ws_port}')
    print(f'[bridge] Open pain_simulator.html?mode=showcase in Chrome/Edge')
    print(f'[bridge] Press Ctrl+C to stop.')

    async with websockets.serve(ws_handler, 'localhost', ws_port):
        await broadcast_loop(queue)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FMS Showcase — Arduino → WebSocket bridge')
    parser.add_argument('--port',    default=None,
                        help='Serial port (e.g. COM3 or /dev/ttyACM0). '
                             'Auto-detected if omitted.')
    parser.add_argument('--ws-port', type=int, default=8765,
                        help='WebSocket server port (default: 8765)')
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    if port is None:
        print('[bridge] ERROR: No serial port found. '
              'Connect the Arduino and retry, or pass --port explicitly.')
        raise SystemExit(1)

    try:
        asyncio.run(main(port, args.ws_port))
    except KeyboardInterrupt:
        print('\n[bridge] Stopped.')
