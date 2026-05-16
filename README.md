# 🧪 VOC Gas Classification from Electronic Nose Sensor Data

> Deep feature extraction from sensor signals + lightweight ML models designed for edge deployment.

---

## 📌 Project Overview

This project develops a machine learning pipeline to classify **Volatile Organic Compounds (VOCs)** using data from an electronic nose (e-nose) sensor array. Raw sensor signals are processed to extract meaningful temporal and peak-based features, which are then used to train and compare three classification models.

The models are built with **edge deployment in mind** — prioritising low complexity and fast inference suitable for microcontrollers or embedded systems, not just cloud servers.

### What is being predicted?
The models predict three targets:
- **Compound** — the exact VOC chemical (e.g. decanal, acetic acid)
- **Functional group** — the chemical family (e.g. Aldehyde, Ester, Phenol)
- **Carbon count** — the molecular size of the compound

---

## 🧬 Dataset & Features

Sensor responses were recorded from multiple sensors across repeated measurements of 10 VOC compounds spanning 5 functional groups.

**Compounds tested:**
| Compound | Functional Group | Carbon Count |
|---|---|---|
| 4-methylphenol, 2-methylphenol, 3-methylphenol | Phenol | 7 |
| Acetic acid | Carboxylic Acid | 2 |
| Butanoic acid | Carboxylic Acid | 4 |
| Decanal, E-2-decenal | Aldehyde | 10 |
| Decan-1-ol | Alcohol | 10 |
| Ethyl decanoate, Decyl acetate | Ester | 12 |

**Extracted features per sensor:**
- `peak_height` — maximum sensor response
- `t_peak_s` — time to peak
- `duration_sec` — signal duration
- `width_half_sec` — signal width at half maximum

Features are pivoted across all sensors to create a flat feature matrix per measurement.

---

## 🤖 Models

Three classifiers were trained and evaluated:

| Model | Split Strategy | Notes |
|---|---|---|
| **Random Forest** | GroupShuffleSplit (70/30) | Fast, interpretable, strong baseline |
| **XGBoost** | GroupShuffleSplit (80/20) | Gradient boosting, tuned for multi-class |
| **Voting Classifier** | GroupShuffleSplit (75/25) | Ensemble of LR + LDA + SVM (soft voting) |

> ⚠️ **Group-aware splitting** is used throughout — measurements from the same compound+sensor run are never split across train and test, preventing data leakage.

---

## 📊 Results

> *(Update this section after running experiments with your actual accuracy scores)*

| Model | Functional Group Acc. | Carbon Count Acc. | Compound Acc. |
|---|---|---|---|
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| Voting Classifier | — | — | — |

---

## 🗂️ Project Structure

```
├── Data_ML.py          # Data loading, feature extraction & pivot table
├── RF.py               # Random Forest classifier
├── XGBoost.py          # XGBoost classifier
├── voting_classifier.py # Voting ensemble classifier
├── requirements.txt    # Python dependencies
└── README.md
```

> Note: Raw CSV data files are not included in this repository. Update the file paths in `Data_ML.py` to point to your local data directory.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Update data paths
In `Data_ML.py`, update the `file_compound_map` dictionary to point to your CSV files:
```python
file_compound_map = {
    "/your/local/path/4methylphenol.csv": "4-methylphenol",
    ...
}
```

### 4. Run a model
```bash
python RF.py
python XGBoost.py
python voting_classifier.py
```

Each script imports the processed data from `Data_ML.py` automatically and outputs a classification report + confusion matrix.

---

## 🔭 Future Work

- [ ] **Hyperparameter tuning** — systematic grid/random search for all three models
- [ ] **Model compression** — quantisation and pruning for edge deployment (TFLite, ONNX)
- [ ] **Real-time inference** — streaming sensor input pipeline for live classification
- [ ] **More compounds** — expand dataset to cover a broader chemical space
- [ ] **Feature selection** — identify the minimal feature set needed for edge accuracy targets
- [ ] **Benchmark on hardware** — test inference latency on a Raspberry Pi or microcontroller

---

## 🛠️ Dependencies

```
scikit-learn
xgboost
pandas
matplotlib
numpy
```

Generate a full pinned list with:
```bash
pip freeze > requirements.txt
```

---

## 👤 Author

**Vijay R. Nair**
- Built as part of research into intelligent sensor systems and edge ML

---

*If you use this work, please consider citing or starring the repository ⭐*
