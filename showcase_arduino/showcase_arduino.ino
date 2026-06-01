/*
 * showcase_arduino.ino
 * FMS Showcase — FSR pressure sensor reader for Arduino Uno R4
 *
 * Reads 6 FSR (Force Sensitive Resistor) sensors on A0–A5.
 * Each FSR is wired as a voltage divider with a 10 kΩ pull-down resistor.
 *
 * Sensor → body location mapping:
 *   A0 = Left arm   A1 = Right arm
 *   A2 = Left leg   A3 = Right leg
 *   A4 = Torso      A5 = Head/neck
 *
 * Every 50 ms, finds the sensor with the highest reading above the noise
 * floor and emits a single JSON line over USB serial at 115200 baud:
 *   {"sensor":N,"pressure":V}
 *
 * If no sensor is active (all below noise floor), nothing is emitted.
 *
 * Arduino Uno R4 note:
 *   The R4 has a 14-bit ADC (0–16383). analogReadResolution(10) in setup()
 *   forces 10-bit output (0–1023), keeping pressure thresholds consistent
 *   with the showcase_bridge.py Python bridge.
 *
 * Wiring (repeat for each of the 6 FSRs):
 *   5V ──── FSR ──── Analog pin (A0–A5)
 *                         │
 *                       10kΩ
 *                         │
 *                        GND
 */

// ── Mode toggle ──────────────────────────────────────────────────────────────
// Comment out PLOTTER_MODE to go back to JSON output for the bridge.
// Uncomment it to see all 6 sensors as live lines in the Serial Plotter.
//#define PLOTTER_MODE

// ── Pin assignments ──────────────────────────────────────────────────────────
const int NUM_SENSORS   = 6;
const int SENSOR_PINS[] = {A0, A1, A2, A3, A4, A5};

// Labels shown in the Serial Plotter legend
const char* SENSOR_NAMES[] = {
    "Head", "Torso", "RightArm", "LeftArm", "RightLeg", "LeftLeg"
};

// ── Threshold ────────────────────────────────────────────────────────────────
// Readings below this are treated as no contact (noise / resting voltage).
// Calibrate after wiring — start here and adjust if sensors trigger at rest.
const int NOISE_FLOOR = 80;

// ── Timing ───────────────────────────────────────────────────────────────────
const unsigned long POLL_INTERVAL_MS = 50;   // 20 Hz polling

unsigned long lastPollTime = 0;

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    // Force 10-bit ADC resolution (0–1023) regardless of R4's 14-bit default.
    // This keeps pressure values consistent with Python bridge thresholds.
    analogReadResolution(10);
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
    unsigned long now = millis();

    if (now - lastPollTime < POLL_INTERVAL_MS) {
        return;
    }
    lastPollTime = now;

    // Find the sensor with the highest reading above the noise floor
    int dominantSensor   = -1;
    int dominantPressure =  0;

    for (int i = 0; i < NUM_SENSORS; i++) {
        int val = analogRead(SENSOR_PINS[i]);
        if (val > NOISE_FLOOR && val > dominantPressure) {
            dominantSensor   = i;
            dominantPressure = val;
        }
    }

#ifdef PLOTTER_MODE
    // Serial Plotter mode — print all 6 sensor values on one line
    // Format: "LeftArm:245 RightArm:0 LeftLeg:0 RightLeg:0 Torso:0 Head:0"
    for (int i = 0; i < NUM_SENSORS; i++) {
        Serial.print(SENSOR_NAMES[i]);
        Serial.print(":");
        Serial.print(analogRead(SENSOR_PINS[i]));
        if (i < NUM_SENSORS - 1) Serial.print(" ");
    }
    Serial.println();
#else
    // JSON mode — used by showcase_bridge.py
    if (dominantSensor >= 0) {
        Serial.print("{\"sensor\":");
        Serial.print(dominantSensor);
        Serial.print(",\"pressure\":");
        Serial.print(dominantPressure);
        Serial.println("}");
    }
#endif
}
