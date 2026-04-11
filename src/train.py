import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8]
    })

    X = df[["x"]]
    y = df["y"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")


if __name__ == "__main__":
    train()
