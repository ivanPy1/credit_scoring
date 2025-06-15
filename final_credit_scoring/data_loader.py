import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE
from features import generate_features

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = generate_features(df)
    X = df.drop(columns=["LoanApproved", "ApplicationDate"])
    y = df["LoanApproved"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)