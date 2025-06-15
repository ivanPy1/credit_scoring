import shap
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data
from model_xgb import build_model
from sklearn.metrics import roc_curve, auc
from xgboost import DMatrix


def main():
    print("[SHAP-XGB] Загрузка данных и обучение модели...")
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train)
    model.fit(X_train, y_train)

    print("[SHAP-XGB] Извлечение обученной модели...")
    booster = model.named_steps["classifier"].get_booster()
    preprocessor = model.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    columns = preprocessor.get_feature_names_out(input_features=X_train.columns)
    X_trans_df = pd.DataFrame(X_train_transformed, columns=columns)

    print("[SHAP-XGB] Инициализация explainer...")
    explainer = shap.Explainer(booster)
    shap_values = explainer(X_trans_df)

    print("[SHAP-XGB] Построение summary-графика...")
    shap.summary_plot(shap_values, X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig("shap_xgb_summary.png")
    print("[SHAP-XGB] SHAP summary plot сохранён в 'shap_xgb_summary.png'")

    print("[ROC-XGB] Построение ROC-кривой...")
    dtest = DMatrix(X_test_transformed)
    y_proba = booster.predict(dtest)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (XGB)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_xgb_curve.png")
    print("[ROC-XGB] ROC-кривая сохранена в 'roc_xgb_curve.png'")

if __name__ == "__main__":
    main()