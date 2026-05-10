from Data_ML import df_pivot
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

