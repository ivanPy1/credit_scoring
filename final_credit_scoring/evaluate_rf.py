from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Model Evaluation:\n")
    print(classification_report(y_test, y_pred, digits=4))

    auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC: {auc:.4f}")