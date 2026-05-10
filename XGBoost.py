import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier

def train_xgb_with_groups(df, target_column):
    start_time = time.time()

    metadata_cols = ["FD_num", "repeat", "compound", "functional_group", "carbon_count", "Sample_Group_ID"]
    X = df.drop(columns=metadata_cols, errors="ignore")
    y = df[target_column]
    groups = df["Sample_Group_ID"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_lbl = le.inverse_transform(y_test)
    y_pred_lbl = le.inverse_transform(y_pred.astype(int))

    print(classification_report(y_test_lbl, y_pred_lbl, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        labels=range(len(le.classes_)),
        display_labels=le.classes_,
        ax=ax,
        xticks_rotation="vertical"
    )
    ax.set_title(f"XGBoost: {target_column}")
    plt.tight_layout()
    plt.show()

    print(f"Time: {time.time() - start_time:.2f} sec")
    return model

model_xgb_fg = train_xgb_with_groups(df_pivot, "functional_group")