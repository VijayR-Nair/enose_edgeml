# VOC Fingerprinting with GC-SOMSA and Machine Learning

> Classifying volatile organic compounds from metal oxide sensor arrays — built lean, built for the edge.

---

## Overview

Conventional gas chromatography (GC) delivers excellent chemical accuracy, but it comes at a cost: lab-grade equipment, trained operators, and no real-world portability. This project takes a different approach.

By coupling a GC column with a compact array of four **Metal Oxide Semiconductor (MOS) sensors** (the GC-SOMSA setup), chemical selectivity is shifted from expensive hardware into software. Raw sensor signals are processed, meaningful peak-based features are extracted, and three supervised ML models are trained to classify VOC gases — all while keeping model complexity deliberately low for **future edge deployment**.

The pipeline was developed as part of a Master's thesis at **Deggendorf Institute of Technology / Fraunhofer Institute**, with practical applicability in areas like packaged food quality monitoring and on-site chemical screening.

---

## The Hardware Setup

```
GC Column  →  MOS Sensor Array (4 layers)  →  Signal Processing  →  ML Classification
              [GC-SOMSA platform]
```

The GC separates compounds chromatographically over time. Four MOS sensor layers at the column outlet each produce a time-series response. Because MOS sensors are non-selective by nature, the intelligence lives in the feature extraction and model — not the sensor itself.

---

## Compounds & Chemical Classes

10 VOCs across 5 functional groups were used:

| Compound | Functional Group | Carbon Count |
|---|---|---|
| 2-, 3-, 4-methylphenol | Phenol | C7 |
| Acetic acid | Carboxylic Acid | C2 |
| Butanoic acid | Carboxylic Acid | C4 |
| Decanal, E-2-decenal | Aldehyde | C10 |
| Decan-1-ol | Alcohol | C10 |
| Ethyl decanoate, Decyl acetate | Ester | C12 |

---

## Signal Processing & Feature Engineering

Raw MOS sensor traces go through a dedicated preprocessing pipeline before any ML:

1. **Butterworth smoothing** — removes high-frequency noise while preserving peak shape
2. **Baseline correction & trimming** — suppresses solvent dominance and drift
3. **Peak detection** — uses Topographic Prominence + Median Absolute Deviation (MAD) for robust, noise-aware peak selection, with fallback heuristics for weak-response sensors
4. **Feature extraction per sensor layer:**

| Feature | Description |
|---|---|
| `peak_height` | Maximum sensor response amplitude |
| `t_peak_s` | Retention time (time to peak) |
| `duration_sec` | Total signal duration |
| `width_half_sec` | Peak width at half maximum |

5. **Feature pivoting** — all per-sensor features are flattened into a single row per measurement, creating a structured feature matrix ready for ML

> Key insight: peak height alone is insufficient for separating structurally similar VOCs (e.g. positional isomers). Adding timing and shape descriptors significantly improves discrimination.

---

## Models

Three classifiers are trained and compared across three classification targets: **compound identity**, **functional group**, and **carbon chain length**.

| Model | Split Strategy | Why it's here |
|---|---|---|
| **Random Forest** | GroupShuffleSplit 70/30 | Fast, interpretable, strong baseline; best compound-level stability |
| **XGBoost** | GroupShuffleSplit 80/20 | Gradient boosting; tuned for multi-class with low depth |
| **Voting Classifier** | GroupShuffleSplit 75/25 | Soft ensemble of LR + LDA + SVM — captures different decision boundaries |

**Group-aware splitting** is used throughout: measurements from the same compound and dilution level are never split across train and test. This prevents data leakage and reflects real-world batch variability.

---


---

## Project Structure

```
.
├── configs/
│   ├── config.yaml
│   └── compound_map.csv
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── src/
│   ├── data_processing/
│   │   ├── gc_sensor_pipeline.py
│   │   └── smooth_extract_features.py
│   └── ml/
│       ├── dataset.py
│       ├── model_factory.py
│       ├── train_models.py
│       └── predict.py
├── outputs/
│   ├── models/
│   ├── reports/
│   └── predictions/
└── README.md
```

---

Pipeline
1. Data Processing

src/data_processing/gc_sensor_pipeline.py

Converts raw sensor .h5 files to .npy
Converts GC .txt files to .npy
Synchronizes GC windows with MOS sensor timestamps
Saves synchronized sensor segments
2. Feature Extraction

src/data_processing/smooth_extract_features.py

Applies Butterworth smoothing
Detects peaks
Estimates baseline
Extracts peak-based features such as height, area, duration, width, asymmetry, and peak timing
3. Model Training

src/ml/train_models.py

Supported models:

Random Forest
XGBoost
Voting Classifier

4. Prediction

src/ml/predict.py

Configuration

Labels are managed through:

configs/compound_map.csv

## Edge Deployment — The Goal

All design decisions in this project point toward one long-term objective: **running VOC classification on a resource-constrained edge device**, without a lab PC or cloud connection.

This influenced every choice made:
- **Feature selection** — only 4 interpretable peak features per sensor; no deep embeddings
- **Model choice** — tree-based and linear models with low inference cost
- **Low model complexity** — XGBoost capped at `max_depth=6`, RF at 100 estimators

Data Policy

Large files are not tracked in GitHub.

Ignored files include:

Raw sensor data
Processed .npy files
Feature CSVs
Trained models
Output reports
Python cache files

---

## Dependencies

```
scikit-learn
xgboost
pandas
matplotlib
numpy
```

Generate a pinned list with:
```bash
pip freeze > requirements.txt
```

---

## About

Developed as part of a Master's thesis:
**"Fingerprinting of VOC Profiles for GC-Sensor Systems"**
Deggendorf Institute of Technology — M.Eng. Artificial Intelligence for Smart Sensors and Actuators
Fraunhofer Institute, Cham, 2025

**Author:** Vijay Radhakrishnan Nair

---

*If this work is useful to you, a ⭐ on the repo goes a long way.*
