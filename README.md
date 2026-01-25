# SAIM: Systemic Attractor Instability Metric (v1.0)

This repository contains the official Python implementation of the **SAIM Analysis Pipeline v1.0** and the **Generative Model Simulation**, as described in the paper:

> **"Informational Perturbation Resolves Precision Collapse and Restores Adaptive Neural Dynamics"**
> *Takafumi Shiga* (2026)

## Overview
This repository provides two key components of the study:

1.  **Empirical Analysis Pipeline (`SAIM_pipeline_v1.0.py`)**:
    Processes raw EEG and fNIRS signals (Muse S Gen 2 Athena) to compute the **Free Energy Proxy (F)**, **Hemodynamic Coupling (HEMO)**, and associated metrics. Corresponds to **Supplementary Text S9**.

2.  **Generative Model Simulation (`simulation.py`)**:
    Reproduces the computational dynamics of Proprioceptive Prediction Error Neglect (PPEN) and the phase transition mechanism induced by Specific Informational Perturbation (SIP). Corresponds to **Supplementary Text S8**.

## Requirements
- Python 3.8+
- Dependencies:
  - `pandas>=1.3.0`
  - `numpy>=1.21.0`
  - `matplotlib>=3.4.0`
  - `seaborn>=0.11.0`
  - `scipy>=1.7.0`
  - `scikit-learn>=0.24.0`

## Usage

### 1. Empirical Data Analysis (S9)
To analyze raw physiological data:
1.  Place your raw CSV data files (Muse S format) in the `data/` directory or the same directory as the script.
2.  Run the pipeline:
    ```bash
    python SAIM_pipeline_v1.0.py
    ```
3.  **Outputs**:
    - `*_TimeSeries.csv`: Frame-by-frame metric calculations.
    - `*_Overall_Stats.csv`: Session-level statistics.
    - `*_Continuous_Dynamics.png`: Visualization of the phase transition.
    - `*_OmniPanel.png`: The 20-panel spectral profiling plot (Figure 2).

### 2. Computational Simulation (S8)
To reproduce the theoretical bistable dynamics and SIP mechanism (Figure 1):
1.  Run the simulation script:
    ```bash
    python simulation.py
    ```
2.  **Outputs**:
    - Displays/Saves the "Phase Transition via SIP" plot, illustrating the transition from the Frozen Attractor to the Adaptive State.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
