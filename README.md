# SAIM: Systemic Attractor Instability Metric (v1.0)

This repository contains the official Python implementation of the **SAIM Analysis Pipeline v1.0**, as described in the paper:

> **"Informational Perturbation Resolves Precision Collapse and Restores Adaptive Neural Dynamics"**
> *Takafumi Shiga* (2026)

## Overview
This software processes raw EEG and fNIRS signals (specifically from the Muse S Gen 2 Athena device) to compute the **Free Energy Proxy (F)** and associated metrics based on the Free Energy Principle.

It features:
- **Dual-wavelength fNIRS analysis** (730nm/850nm) with ambient light correction.
- **Precision-Weighted Prediction Error (PE)** calculation.
- **Systemic Phase Transition** detection.

This codebase corresponds strictly to the methods defined in **Supplementary Text S9** of the manuscript.

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

1. **Setup**: Place your raw CSV data files (Muse S format) in the `data/` directory or the same directory as the script.

2. **Run Analysis**:
   Execute the pipeline to generate TimeSeries and Statistical outputs:
   ```bash
   python SAIM_pipeline_v1.0.py

3.　**Outputs**:
The script will automatically generate:
* `*_TimeSeries.csv`: Frame-by-frame metric calculations.
* `*_Overall_Stats.csv`: Session-level statistics.
* `*_Continuous_Dynamics.png`: Visualization of the phase transition.
* `*_OmniPanel.png`: The 20-panel spectral profiling plot used in Figure 2.



## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

