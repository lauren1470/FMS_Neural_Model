# FMS Neural Model

A spiking neural network (SNN) model of the spinal dorsal horn gate control circuit, used to simulate fibromyalgia (FMS) pathology and train machine learning classifiers to distinguish healthy from FMS firing patterns.

Built with [Brian2](https://brian2.readthedocs.io/) and scikit-learn.

---

## Overview

Fibromyalgia is characterised by central sensitisation in the spinal dorsal horn: enhanced NMDA receptor activity drives wind-up (progressive amplification of WDR neuron firing), while reduced GABAergic inhibition impairs the gate control mechanism that normally limits pain signalling.

This project models that circuit computationally, generates a labelled dataset of simulated spike trains across 1000 trials, extracts neurophysiological features, and trains binary classifiers to separate healthy and FMS states with high accuracy.

---

## Project Structure

```
FMS_Neural_Model/
|
|-- src/
|   |-- neurons.py          # LIF neuron parameters (WDR, GABA, C-fibre, Ab-fibre)
|   |-- synapses.py         # Synapse equations and pathology weight configurations
|   |-- stimulation.py      # Stimulation protocols (constant, ramp, burst, mixed)
|   |-- simulations.py      # Brian2 simulation runner and figure generation
|   |-- features.py         # Spike-train feature extraction (19 features)
|   |-- generate_dataset.py # Automated dataset generation across N trials
|   |-- classifier.py       # ML training pipeline (Random Forest + SVM)
|
|-- data/
|   |-- dataset.csv         # 1000-trial labelled dataset (500 healthy / 500 FMS)
|
|-- results/
|   |-- initial/            # Classifier results (default hyperparameters)
|   |-- tuned/              # Classifier results (grid-search tuned hyperparameters)
|
|-- models/
|   |-- initial/            # Saved model files (initial run)
|   |-- tuned/              # Saved model files (tuned run)
|
|-- figures/
|   |-- simulations/        # Raster plots and voltage traces per state
|   |-- initial/            # Confusion matrices and feature importance (initial)
|   |-- tuned/              # Confusion matrices and feature importance (tuned)
|
|-- requirements.txt
|-- README.md
```

---

## Circuit Model

The model implements a gate control circuit with four neuron populations:

| Population | Count | Role |
|---|---|---|
| C-fibres | 100 | Nociceptive input (Poisson spike trains) |
| Ab-fibres | 200 | Innocuous tactile input (Poisson spike trains) |
| WDR neurons | 50 | Wide dynamic range projection neurons (output) |
| GABA interneurons | 30 | Inhibitory gate neurons |

**Synaptic pathways:**
- C-fibres -> WDR via AMPA + NMDA (excitatory, enables wind-up)
- C-fibres -> GABA via AMPA
- Ab-fibres -> GABA via AMPA (gate activation)
- GABA -> WDR via GABA-A (inhibitory gate)

**FMS pathology** is modelled by two parameter changes:
- NMDA weight: 1x (healthy) -> 3x (FMS) — enhanced wind-up
- GABA weight: 1x (healthy) -> 0.4x (FMS) — 60% gate reduction

An **intervention state** is also available (interpolated between FMS and healthy), used for demonstration purposes only and not included in classifier training.

---

## Setup

### Requirements

- Python 3.10+
- Windows (tested), Linux/macOS should work

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/lauren1470/FMS_Neural_Model.git
   cd FMS_Neural_Model
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv fms_env
   fms_env\Scripts\activate        # Windows
   source fms_env/bin/activate     # Linux/macOS
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

> **Note:** Brian2 runs in numpy fallback mode on Windows without a C compiler installed. This produces identical results but is slower (~2.5 hours for 1000 trials). To enable Cython compilation, install Visual Studio Build Tools with the "Desktop development with C++" workload.

---

## Usage

### 1. Run visualisation simulations

Generates raster plots and voltage traces for all three states (healthy, FMS, intervention) and saves them to `figures/simulations/`.

```
python src/simulations.py
```

### 2. Generate the dataset

Runs 1000 simulation trials (500 healthy, 500 FMS) across four stimulation protocols with randomised parameters, extracts features, and saves a CSV file.

```
python src/generate_dataset.py
```

Options:
```
--n_trials   Total number of trials (default: 1000)
--duration   Simulation duration in ms (default: 1000)
--output     Output CSV path (default: dataset.csv)
--seed       Base random seed (default: 42)
```

### 3. Train classifiers

Trains Random Forest and SVM (RBF kernel) classifiers on the dataset and saves results, figures, and model files.

Default hyperparameters:
```
python src/classifier.py --data data/dataset.csv
```

With grid-search hyperparameter tuning:
```
python src/classifier.py --data data/dataset.csv --tune
```

---

## Features

19 spike-train features are extracted per trial, grouped into five categories:

| Category | Features |
|---|---|
| Firing rate | `wdr_mean_rate`, `gaba_mean_rate`, `wdr_peak_rate` |
| Spike timing | `wdr_isi_mean`, `wdr_isi_std`, `wdr_isi_cv`, `gaba_isi_mean`, `gaba_isi_std`, `gaba_isi_cv` |
| Wind-up | `wdr_windup_ratio`, `wdr_evoked_response` |
| Temporal | `wdr_early_rate`, `wdr_late_rate` |
| Population | `wdr_burst_count`, `wdr_burst_fraction`, `ei_ratio`, `wdr_active_fraction`, `wdr_total_spikes`, `gaba_total_spikes` |

---

## Results

All results are from the full 1000-trial dataset (500 healthy / 500 FMS, 70/30 stratified split).

### Classifier performance (tuned)

| Model | Test Accuracy | CV Accuracy |
|---|---|---|
| Random Forest | 100% | 100% |
| SVM (RBF) | 100% | 99.71% (+/- 0.57%) |

### Top features by importance (Random Forest)

| Feature | Importance |
|---|---|
| `wdr_windup_ratio` | 33.6% |
| `wdr_evoked_response` | 20.9% |
| `wdr_mean_rate` | 8.3% |
| `wdr_burst_count` | 9.1% |

### State separation

| Metric | Healthy | FMS |
|---|---|---|
| WDR mean rate | 1.67 Hz | 48.37 Hz |
| Wind-up ratio | 0.008 | 5.03 |
| WDR neurons active | ~41% | ~100% |

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `tau_m` (WDR) | 2.84 ms | Membrane time constant |
| `tau_m` (GABA) | 2.84 ms | Membrane time constant |
| AMPA decay | 6 ms | Fast excitatory kinetics |
| NMDA decay | 170 ms | Slow excitatory (wind-up) |
| GABA-A decay | 20 ms | Inhibitory kinetics |
| GAIN | 5.5 | Global synaptic scaling factor |
| NMDA weight (FMS) | 3x healthy | Central sensitisation |
| GABA weight (FMS) | 0.4x healthy | Gate failure |

---

## Stimulation Protocols

Four protocols are used during dataset generation, each with randomised parameters per trial:

| Protocol | Description |
|---|---|
| `constant` | Sustained C-fibre input at fixed rate |
| `ramp` | C-fibre rate increasing linearly over the trial |
| `burst` | Periodic high-frequency bursts against a low baseline |
| `mixed` | Simultaneous elevated C-fibre and Ab-fibre input |

---

## Limitations

- Single-compartment LIF neurons do not capture dendritic processing
- Poisson input does not model peripheral receptor adaptation
- GAIN=5.5 is a pragmatic scaling factor, not derived from a specific physiological measurement
- tau_m=2.84 ms is shorter than typical spinal neuron time constants (~10-20 ms); this reflects the compressed simulation timescale
- The healthy state produces WDR=0 in many trials due to strong stochastic GABA activation — this is biologically interpretable as shunting inhibition but represents an idealised gate
- GABA mean firing rate differs modestly between states (healthy: ~70 Hz, FMS: ~66 Hz); the primary discriminating signal comes from WDR activity

---

## Dependencies

| Package | Version |
|---|---|
| Brian2 | 2.5.4 |
| NumPy | 1.26.4 |
| pandas | 2.1.4 |
| scikit-learn | 1.3.2 |
| matplotlib | 3.8.0 |
| seaborn | 0.13.2 |
| scipy | 1.11.4 |
| joblib | 1.5.3 |
| Cython | 3.2.4 |
