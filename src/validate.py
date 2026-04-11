import pandas as pd


def validate():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert not df.isnull().values.any()


if __name__ == "__main__":
    validate()
