from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import pandas as pd
from xgboost import XGBClassifier


def build_model(X):
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric),
        ("cat", categorical_transformer, categorical)
    ])

    xgb_clf = XGBClassifier(
        device='cuda',
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        verbosity=1,
        colsample_bytree=0.6,
        gamma=0,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=500,
        reg_lambda=0,
        subsample=0.8
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb_clf)
    ])

    return pipeline
