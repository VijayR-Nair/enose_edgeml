from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def build_model(model_name, random_state=42):
    """
    Create a machine learning model by name.

    Supported models:
    - rf
    - xgboost
    - voting
    """

    model_name = model_name.lower()

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
    )

    if model_name == "rf":
        return rf

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="mlogloss",
        )

    if model_name == "voting":
        if XGBClassifier is None:
            raise ImportError("xgboost is required for the voting model.")

        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="mlogloss",
        )

        return VotingClassifier(
            estimators=[
                ("rf", rf),
                ("xgb", xgb),
            ],
            voting="soft",
        )

    raise ValueError(f"Unknown model name: {model_name}")