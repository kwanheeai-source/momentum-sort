import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent


def download_all():
    DATA_DIR.mkdir(exist_ok=True)

    # Abalone
    abalone_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    pd.read_csv(abalone_url, header=None).to_csv(DATA_DIR / "abalone.data", index=False)

    # Wine Quality
    wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    pd.read_csv(wine_url, sep=';').to_csv(DATA_DIR / "winequality-red.csv", index=False)

    # California Housing (MedInc)
    cali_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    pd.read_csv(cali_url)[["median_income"]].to_csv(DATA_DIR / "california_medinc.csv", index=False)

    print("✅ All datasets downloaded to ./data/")


if __name__ == "__main__":
    download_all()