
import pandas as pd

FEATURES_TO_USE = ["peak_height", "t_peak_s", "duration_sec", "width_half_sec"]

file_compound_map = {
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_4Methylphenol.csv": "4-methylphenol",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_2methylphenol.csv": "2-methylphenol",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_3methylphenol.csv": "3-methylphenol",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_acetic.csv": "acetic acid",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_butanoic.csv": "butanoic acid",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_decanal.csv": "decanal",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_E2Decanal.csv": "E-2-decenal",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_decanol.csv": "decan-1-ol",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_Ethyldecanoate.csv": "ethyl decanoate",
    "/home/vijayrnair/IVV_JSON/Data Pro/08|12_alldata/gcmetric_all_decylacetate.csv": "decyl acetate",
}

compound_properties_map = {
    "4-methylphenol": {"functional_group": "Phenol", "carbon_count": 7},
    "2-methylphenol": {"functional_group": "Phenol", "carbon_count": 7},
    "3-methylphenol": {"functional_group": "Phenol", "carbon_count": 7},
    "acetic acid":    {"functional_group": "Carboxylic Acid", "carbon_count": 2},
    "butanoic acid":  {"functional_group": "Carboxylic Acid", "carbon_count": 4},
    "decanal":        {"functional_group": "Aldehyde", "carbon_count": 10},
    "E-2-decenal":    {"functional_group": "Aldehyde", "carbon_count": 10},
    "decan-1-ol":     {"functional_group": "Alcohol", "carbon_count": 10},
    "ethyl decanoate":{"functional_group": "Ester", "carbon_count": 12},
    "decyl acetate":  {"functional_group": "Ester", "carbon_count": 12},
}

df_list = []
for file, compound in file_compound_map.items():
    try:
        df = pd.read_csv(file)

        if "FD" in df.columns and "FD_num" not in df.columns:
            df = df.rename(columns={"FD": "FD_num"})
        if "t_lb_s" in df.columns and "t_base_s" not in df.columns:
            df = df.rename(columns={"t_lb_s": "t_base_s"})
        if "FD_num" in df.columns:
            df["FD_num"] = df["FD_num"].astype(str)

        df["compound"] = compound
        df["functional_group"] = compound_properties_map[compound]["functional_group"]
        df["carbon_count"] = compound_properties_map[compound]["carbon_count"]

        df_list.append(df)
    except FileNotFoundError:
        pass

if not df_list:
    raise ValueError("No data loaded. Check file paths.")

df_combined = pd.concat(df_list, ignore_index=True)

if "repeat" not in df_combined.columns:
    df_combined["repeat"] = df_combined.groupby(["compound", "FD_num", "sensor"]).cumcount() + 1

available_features = [f for f in FEATURES_TO_USE if f in df_combined.columns]

index_cols = ["FD_num", "repeat", "compound", "functional_group", "carbon_count"]
df_pivot = df_combined.pivot_table(index=index_cols, columns="sensor", values=available_features)

df_pivot.columns = ["_".join(map(str, col)).strip() for col in df_pivot.columns.values]
df_pivot = df_pivot.reset_index().fillna(0)

df_pivot["Sample_Group_ID"] = df_pivot["compound"] + "_" + df_pivot["FD_num"].astype(str)

import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def train_rf_with_groups(df, target_column):
    metadata_cols = ["FD_num", "repeat", "compound", "functional_group", "carbon_count", "Sample_Group_ID"]
    X = df.drop(columns=metadata_cols, errors="ignore")
    y = df[target_column]
    groups = df["Sample_Group_ID"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy ({target_column}): {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, xticks_rotation="vertical")
    ax.set_title(f"Random Forest: {target_column}")
    plt.tight_layout()
    plt.show()

    return model

model_compound = train_rf_with_groups(df_pivot, "compound")
model_fg = train_rf_with_groups(df_pivot, "functional_group")