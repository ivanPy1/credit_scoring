import shap
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data
from model_rf import build_model
from sklearn.metrics import roc_curve, auc


def main():
    print("[SHAP-RF] Загрузка данных и обучение модели...")
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train)
    model.fit(X_train, y_train)

    print("[SHAP-RF] Извлечение обученной модели...")
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    columns = preprocessor.get_feature_names_out(input_features=X_train.columns)
    X_trans_df = pd.DataFrame(X_train_transformed, columns=columns)

    print("[SHAP-RF] Инициализация explainer...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_trans_df)

    print("[SHAP-RF] Проверка форм:")
    print("shap_values[:, :, 1].shape:", shap_values[:, :, 1].shape)
    print("X_trans_df.shape:", X_trans_df.shape)

    print("[SHAP-RF] Построение summary-графика...")
    shap.summary_plot(shap_values[:, :, 1], X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig("shap_rf_summary.png")
    print("[SHAP-RF] SHAP summary plot сохранён в 'shap_rf_summary.png'")

    print("[ROC-RF] Построение ROC-кривой...")
    y_proba = classifier.predict_proba(X_test_transformed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (RandomForest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_rf_curve.png")
    print("[ROC-RF] ROC-кривая сохранена в 'roc_rf_curve.png'")


if __name__ == "__main__":
    main()