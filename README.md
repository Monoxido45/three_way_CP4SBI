# Three Way Calibration Epistemic Uncertainty Quantification on Conformal Prediction for Simulation Based Inference (CP4SBI)

## Summary

This package implements an additional three-way approach to quantify calibration epistemic uncertainty in conformal prediction for simulation‑based inference (CP4SBI) methods. It provides tools to:
- Quantify epistemic uncertainty arising from limited calibration data.
- Produce three types of predictive regions: inside, outside or undetermined.
- Controled probabilities of error for inside and outside regions.

## Reproducing the paper results

All code, configuration files, and checkpoints needed to reproduce the paper's results, figures, and tables are provided in the Experiments folder. After installing the package, run the scripts in Experiments (see the script and config filenames for the intended pipelines); configuration files and inline comments in those scripts describe the exact command lines and options used for the reported runs.

---

## Installation

### Using pip

```bash
pip install .
```

### Using Conda

```bash
conda create -n tw_CP4SBI_env python=3.9
conda activate tw_CP4SBI_env
pip install .
```

---







