def generate_features(df):
    df = df.copy()
    df["AssetHealth"] = df["NetWorth"] / (df["TotalAssets"] + 1)
    df.drop(columns=["NetWorth", "TotalAssets", "AnnualIncome"], inplace=True)
    return df