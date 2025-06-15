from data_loader import load_data
from model_xgb import build_model
from evaluate_xgb import evaluate_model

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
