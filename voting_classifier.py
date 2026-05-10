import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def train_vc_with_groups(df, target_column):
    metadata_cols = ["FD_num", "repeat", "compound", "functional_group", "carbon_count", "Sample_Group_ID"]
    X = df.drop(columns=metadata_cols, errors="ignore")

    y = df[target_column]
    groups = df["Sample_Group_ID"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    clf1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf2 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    clf3 = make_pipeline(StandardScaler(), SVC(probability=True, kernel="rbf"))

    model = VotingClassifier(
        estimators=[("lr", clf1), ("lda", clf2), ("svm", clf3)],
        voting="soft"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy ({target_column}): {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        labels=range(len(le.classes_)),
        display_labels=le.classes_,
        ax=ax,
        xticks_rotation="vertical"
    )
    ax.set_title(f"Voting Classifier: {target_column}")
    plt.tight_layout()
    plt.show()

    return model

model_vc_fg = train_vc_with_groups(df_pivot, "functional_group")
model_vc_cc = train_vc_with_groups(df_pivot, "carbon_count")
model_vc_compound = train_vc_with_groups(df_pivot, "compound")