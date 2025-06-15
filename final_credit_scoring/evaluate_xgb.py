from sklearn.metrics import classification_report, roc_auc_score
from xgboost import DMatrix, XGBClassifier

def evaluate_model(model, X_test, y_test):
    print("Model Evaluation:\n")

    try:
        booster = model.named_steps["classifier"].get_booster()
        preprocessor = model.named_steps["preprocessor"]
        X_test_transformed = preprocessor.transform(X_test)

        dtest = DMatrix(X_test_transformed, nthread=-1)

        y_pred_proba = booster.predict(dtest)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        print("[GPU] Предсказание через booster (GPU)")
        print(classification_report(y_test, y_pred, digits=4))
        print(f"ROC AUC (proba): {roc_auc_score(y_test, y_pred_proba):.4f}")

    except Exception as e:
        print(f"[CPU] Fallback: {e}")
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred_proba = None

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, digits=4))

        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC AUC (proba): {auc:.4f}")
        else:
            auc = roc_auc_score(y_test, y_pred)
            print(f"ROC AUC (labels): {auc:.4f}")

