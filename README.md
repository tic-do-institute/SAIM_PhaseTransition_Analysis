# SAIM: Systemic Attractor Instability Metric (v1.0)

This repository contains the official Python implementation of the **SAIM Analysis Pipeline v1.0** and the **Generative Model Simulation**, as described in the manuscript:

> **"Informational Perturbation Resolves Precision Collapse and Restores Adaptive Neural Dynamics"**
> *Takafumi Shiga* 

---

## ⚠️ Important Note for Reviewers

**The analysis logic and preprocessing pipeline presented in this repository were fixed prior to the commencement of the study (pre-registered) and have NOT been modified based on post-hoc observations.**

While we acknowledge the existence of advanced artifact removal techniques (e.g., ASR, CCA, or spectral parameterization), we prioritized **transparency, reproducibility, and minimal signal distortion** for this mechanistic investigation. The validity of this "fixed pipeline" approach is supported by rigorous experimental constraints and control analyses (detailed in the manuscript's *Supplementary Information, Text S2 & S3*).

---

## Overview

This repository provides the three core components of the study:

1. **Calibration & Noise Validation (`calibration_analysis.py`)**:
   A strict quality control script that verifies the separation between neural signals (EEG) and myogenic artifacts (EMG) using a standardized "Standing Calibration Protocol".
   * *Ensures that the portable EEG device properly distinguishes brain activity from muscle noise.*

2. **Empirical Analysis Pipeline (`SAIM_pipeline_v1.0.py`)**:
   Processes raw EEG and fNIRS signals (Muse S Gen 2 Athena) to compute the **Free Energy Proxy (F)**, **Hemodynamic Coupling (HEMO)**, and associated metrics. It implements the automated detection of **PPEN** (Proprioceptive Prediction Error Neglect), including the "Paradoxical Rigidity" subtype (Rule C).
   * *Corresponds to Supplementary Text S9.*

3. **Generative Model Simulation (`simulation.py`)**:
   Reproduces the computational dynamics of the "Frozen Attractor" and the phase transition mechanism induced by Specific Informational Perturbation (SIP).
   * *Corresponds to Supplementary Text S8.*

---

## 1. Calibration & Noise Validation (Quality Control)

To guarantee signal validity, all subjects underwent a strict **Standing Calibration Protocol** prior to the main experiment. The `calibration_analysis.py` script automatically verifies the signal quality based on the following logic.

### Protocol
* **Rest (0-55s):** Standing, Eyes Closed. (Baseline EEG)
* **Tasks (After 59s):**
  1. **Jaw Clench:** Standing, Eyes Closed. (Temporalis muscle activation)
  2. **Eyebrow Raise:** Standing, **Eyes Open**. (Maximized Frontalis muscle activation)
  3. **Blink:** Standing, Natural. (EOG artifact)

### Analysis Logic (`calibration_analysis.py`)
The script employs a **Hybrid Detection Algorithm**:
* **Reference Standard (S001):** Uses medically validated, manually fixed timing windows to serve as the ground truth for signal separation.
* **Automated Screening (S002+):** Applies a "59-second skip rule" followed by a sequential peak detection algorithm (Jaw → Eye → Blink) to automatically validate each subject.
* **Pass Criteria:** A significant difference in spectral slope and linear power between Rest and Jaw phases (Diff > 0.3) is required for inclusion.

---

## 2. Main Analysis Logic (`SAIM_pipeline_v1.0.py`)

This pipeline processes raw physiological data to quantify the "Frozen Attractor" state. The core algorithms are defined as follows:

### A. Free Energy Proxy (F)
We utilize **Gamma Band Power (30-45 Hz)** as a macro-scale proxy for prediction error (PE) signaling, based on predictive coding frameworks (Friston et al.).
* **Preprocessing:** Short-time Fourier Transform (STFT) with a 1-second Hanning window.
* **Metric:** $F(t) = \log(P_{\gamma}(t) / P_{baseline})$
* **Rationale:** Gamma oscillations correlate with bottom-up prediction error propagation.

### B. Hemodynamic Coupling (HEMO)
We quantify the stability of neurovascular coupling using raw infrared (IR) and red light photodiode signals.
* **Metric:** Variance of the raw PPG signal over a sliding window.
* **Interpretation:** High variance indicates robust hemodynamic responsiveness; low variance (flatline) indicates "Hemodynamic Uncoupling" or stress-induced vasoconstriction.

### C. PPEN Detection (Rule C: Paradoxical Rigidity)
The pipeline automatically detects "Proprioceptive Prediction Error Neglect" (PPEN) when the system enters a specific state of high-rigidity but low-adaptability.

**Detection Logic:**
A time window is flagged as **PPEN (Frozen State)** if:
1. **High Gamma (F):** The system is generating excessive prediction errors (Hyper-priors).
2. **Low HEMO:** Despite high electrical activity, hemodynamic supply is suppressed (Uncoupling).
3. **Low Beta:** Lack of top-down motor suppression.

> *Note: This "Paradoxical Rigidity" index is the primary biomarker for the "Frozen Attractor" discussed in the manuscript.*

---

## Directory Structure


```

.
├── data/
│   ├── calibration/
│   │   └── S001_Calibration.csv  # Sample validated data for demo
│   └── (Main experiment data not included for privacy)
├── docs/
│   └── Muse_Technical_Validation.pdf
├── output/                       # Generated CSVs and PNGs
├── calibration_analysis.py       # Quality Control Script (Hybrid Logic)
├── SAIM_pipeline_v1.0.py         # Main Analysis Script (F, HEMO, PPEN)
├── simulation.py                 # Generative Model Script
├── LICENSE
└── README.md

```

---

## Usage

### 1. Calibration Check (Quality Control)
To verify the signal quality of the sample data (or new subjects):
```bash
python calibration_analysis.py

```

* **Output:** Prints the `Status` (OK/CHECK) and spectral metrics for each subject.

### 2. Empirical Data Analysis (S9)

**Note:** Due to privacy protection protocols for human subjects, the full dataset (N=106) is not included in this repository. A validated sample file (`S001_Calibration.csv`) is provided for demonstration purposes.

To analyze the data:

1. **For Demo:** The script will automatically detect the sample file in `data/`.
2. **For Your Own Data:** Place your Muse CSV files in the `data/` directory.
3. Run the pipeline:
```bash
python SAIM_pipeline_v1.0.py

```



* **Outputs**:
* `*_TimeSeries.csv`: Frame-by-frame metric calculations.
* `*_Overall_Stats.csv`: Session-level statistics (Pre vs Post).
* `*_Continuous_Dynamics.png`: Visualization of the phase transition.
* `*_OmniPanel.png`: The 20-panel spectral profiling plot (Figure 2).



### 3. Computational Simulation (S8)

To reproduce the theoretical bistable dynamics and SIP mechanism (Figure 1):

```bash
python simulation.py

```

---

## Data Availability

The **source code** and a **representative sample dataset (S001)** are available in this repository to ensure the reproducibility of the analysis pipeline.

The full dataset collected from human subjects (N=106) is not publicly available due to ethical restrictions and privacy protection protocols regarding clinical data. However, the de-identified dataset may be available from the corresponding author (*Takafumi Shiga*) upon reasonable request and subject to ethical approval.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=./LICENSE) file for details.

---

Copyright (c) 2026 TIC-DO Institute.

```

```
