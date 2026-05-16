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

## Results

> *(Fill in your actual accuracy values after running the models)*

**Peak height features only:**

| Model | Functional Group | Carbon Count | Compound |
|---|---|---|---|
| Random Forest | 59.26 | 58.02 | 32.10 |
| XGBoost | 59.00 | 48.00 | 28.00 |
| Voting Classifier | 65.26 | 65.32 | 50.72 |

**Integrated feature set (height + timing + shape):**

| Model | Functional Group | Carbon Count | Compound |
|---|---|---|---|
| Random Forest | 100.00 | 98.77 | 92.59 |
| XGBoost | 96.00 | 96.00  | 91.00 |
| Voting Classifier | 100.00 | 100.00 | 81.16 |

---

## Project Structure

```
├── Data_ML.py              # Data loading, preprocessing & feature pivot table
├── RF.py                   # Random Forest classifier
├── XGBoost.py              # XGBoost classifier
├── voting_classifier.py    # Soft voting ensemble (LR + LDA + SVM)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your data paths
In `Data_ML.py`, update `file_compound_map` to point to your local CSV files:
```python
file_compound_map = {
    "/your/path/to/4methylphenol.csv": "4-methylphenol",
    ...
}
```

### 4. Run a model
```bash
python RF.py
python XGBoost.py
python voting_classifier.py
```

Each script loads data from `Data_ML.py` and outputs a classification report and confusion matrix.

---

## Edge Deployment — The Goal

All design decisions in this project point toward one long-term objective: **running VOC classification on a resource-constrained edge device**, without a lab PC or cloud connection.

This influenced every choice made:
- **Feature selection** — only 4 interpretable peak features per sensor; no deep embeddings
- **Model choice** — tree-based and linear models with low inference cost
- **Low model complexity** — XGBoost capped at `max_depth=6`, RF at 100 estimators

### Roadmap toward edge deployment

- [ ] **Model export** — convert trained models to ONNX or TFLite for embedded inference
- [ ] **Quantisation & pruning** — reduce model size for microcontroller memory constraints
- [ ] **Real-time streaming pipeline** — trigger classification from live sensor input without GC timing reference
- [ ] **GC-independent mode** — learn VOC fingerprints directly from MOS time-series, removing the dependency on chromatographic separation entirely
- [ ] **Retention-time normalisation** — use internal standards to compensate for run-to-run shifts in field conditions
- [ ] **Robustness testing** — evaluate performance under humidity variation, sensor drift, and ambient background VOCs
- [ ] **Hardware benchmark** — measure inference latency on Raspberry Pi or STM32 microcontroller
- [ ] **Expand compound library** — broader chemical space for more generalizable models

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
