# VOC Fingerprinting with GC-SOMSA and Machine Learning

> Lightweight classification of volatile organic compounds using GC-separated MOS sensor signals and edge-ready machine learning.

---

## Overview

Conventional gas chromatography delivers strong chemical separation, but typical GC systems are expensive, lab-bound, and require trained operators. This project explores a leaner alternative: combining a gas chromatography column with a compact four-layer **Metal Oxide Semiconductor Sensor Array (MOSSA)** and using machine learning to classify volatile organic compounds (VOCs).

The core idea is to shift part of the selectivity problem from hardware into software. Instead of relying on complex analytical instrumentation alone, the pipeline preprocesses raw sensor traces, extracts interpretable peak-based features, and trains compact supervised models for VOC classification.

The work was developed as part of the Master's thesis **“Fingerprinting of VOC Profiles for GC-Sensor Systems”** at **Deggendorf Institute of Technology / Fraunhofer Institute, Cham**. The long-term goal is practical deployment in constrained environments such as packaged food quality monitoring, on-site screening, and portable chemical sensing.

---

## System Concept

```text
GC Column  →  4-Layer MOS Sensor Array  →  Signal Processing  →  Feature Matrix  →  ML Classification
              GC-SOMSA platform
```

The GC column separates VOCs over time before they reach the sensor array. Each MOS layer records a time-series response at the column outlet. Since MOS sensors are broadly responsive rather than inherently selective, the classification power comes from the combination of chromatographic timing, peak-shape information, and machine learning.

---

## Dataset

The study uses **10 VOCs** from **5 functional groups**.

| Compound | Functional group | Carbon count |
|---|---:|---:|
| 2-methylphenol, 3-methylphenol, 4-methylphenol | Phenol | C7 |
| Acetic acid | Carboxylic acid | C2 |
| Butanoic acid | Carboxylic acid | C4 |
| Decanal, E-2-decenal | Aldehyde | C10 |
| Decan-1-ol | Alcohol | C10 |
| Ethyl decanoate, decyl acetate | Ester | C12 |

The models are trained for three target tasks:

- **Compound identity**
- **Functional group**
- **Carbon chain length**

---

## Signal Processing Pipeline

Raw MOS sensor traces are converted into a structured feature matrix through the following steps:

1. **Data conversion**  
   Raw sensor `.h5` files and GC `.txt` files are converted into NumPy arrays.

2. **Synchronization**  
   GC windows are aligned with MOS sensor timestamps to isolate the relevant response segment for each measurement.

3. **Smoothing**  
   Butterworth filtering removes high-frequency noise while preserving peak shape.

4. **Baseline correction and trimming**  
   Solvent-dominated regions and sensor drift are reduced before feature extraction.

5. **Peak detection**  
   Topographic prominence and Median Absolute Deviation (MAD) are used for robust, noise-aware peak selection. Fallback heuristics handle weak-response sensor layers.

6. **Feature extraction**  
   Peak-based descriptors are extracted from each sensor layer.

7. **Feature pivoting**  
   Per-sensor features are flattened into one row per measurement and used as model input.

### Core Features

| Feature | Description |
|---|---|
| `peak_height` | Maximum response amplitude |
| `t_peak_s` | Retention time / time to peak |
| `duration_sec` | Total response duration |
| `width_half_sec` | Peak width at half maximum |

> Peak height alone is not enough to separate structurally similar VOCs, especially positional isomers. Timing and shape features improve discrimination while keeping the model lightweight.

---

## Machine Learning Models

Three supervised classifiers are implemented and compared.

| Model | Split strategy | Purpose |
|---|---|---|
| Random Forest | GroupShuffleSplit 70/30 | Strong, interpretable baseline with stable compound-level performance |
| XGBoost | GroupShuffleSplit 80/20 | Gradient boosting with controlled tree depth for compact inference |
| Voting Classifier | GroupShuffleSplit 75/25 | Soft ensemble combining linear and kernel-based decision boundaries |

The Voting Classifier combines:

- Logistic Regression
- Linear Discriminant Analysis
- Support Vector Machine

### Leakage Prevention

All experiments use **group-aware splitting**. Measurements from the same compound and dilution level are kept together in either the training set or the test set. This prevents data leakage and gives a more realistic estimate of performance under batch variability.

---

## Edge Deployment Focus

The project is designed with future edge deployment in mind. This affects the entire pipeline:

- **Interpretable features instead of deep embeddings**
- **Small feature vectors: four peak descriptors per sensor layer**
- **Low-complexity models suitable for constrained inference**
- **Random Forest limited to 100 estimators**
- **XGBoost capped at `max_depth=6`**
- **No cloud dependency required for inference**

The intended direction is a portable VOC classification workflow that can run close to the sensor system, without requiring a lab PC or remote server.

---

## Project Structure

```text
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

## Main Modules

### Data Processing

`src/data_processing/gc_sensor_pipeline.py`

- Converts raw MOS sensor `.h5` files to `.npy`
- Converts GC `.txt` files to `.npy`
- Synchronizes GC windows with MOS timestamps
- Saves synchronized sensor segments

### Feature Extraction

`src/data_processing/smooth_extract_features.py`

- Applies Butterworth smoothing
- Performs baseline estimation
- Detects response peaks
- Extracts peak height, area, duration, width, asymmetry, and timing features

### Model Training

`src/ml/train_models.py`

Supported models:

- Random Forest
- XGBoost
- Voting Classifier

### Prediction

`src/ml/predict.py`

- Loads trained models
- Applies the same feature schema used during training
- Generates VOC predictions for new measurements

---

## Configuration

Compound metadata and labels are managed through:

```text
configs/compound_map.csv
```

Project-level settings are stored in:

```text
configs/config.yaml
```

---

## Data Policy

Large and generated files are intentionally excluded from version control.

Ignored files include:

- Raw sensor data
- Processed `.npy` files
- Feature CSV files
- Trained model artifacts
- Output reports
- Prediction files
- Python cache files

This keeps the repository lightweight and focused on reproducible code.

---

## Dependencies

Core dependencies:

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
```

Generate a pinned environment file with:

```bash
pip freeze > requirements.txt
```

---

## Thesis Context

Developed as part of the Master's thesis:

**Fingerprinting of VOC Profiles for GC-Sensor Systems**  
M.Eng. Artificial Intelligence for Smart Sensors and Actuators  
Deggendorf Institute of Technology / Fraunhofer Institute, Cham  
2025

**Author:** Vijay Radhakrishnan Nair

---

## Citation / Acknowledgement

If this project is useful for your work, please consider citing the thesis or giving the repository a ⭐.
