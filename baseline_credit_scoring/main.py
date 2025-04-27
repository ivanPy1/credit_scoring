from data_loader import load_data
from model import build_simple_model
from evaluate import evaluate_model

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = build_simple_model(X_train)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()