import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent


def load_nyc_taxi_skewed_column(year=2024, month=1, n_rows=5_000_000):
    """Downloads/loads one month of NYC taxi data and returns the skewed fare_amount column."""
    # Kaggle parquet (recommended) or official URL
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    print(f"Loading NYC Taxi {year}-{month:02d} (~4M rows)...")

    df = pd.read_parquet(url)
    fares = df["fare_amount"].dropna().values.astype(np.float64)

    # Take first n_rows (or all if smaller)
    fares = fares[:n_rows]
    print(f"Loaded {len(fares):,} skewed values (fare_amount)")

    # Save for benchmarks
    np.save(DATA_DIR / f"nyc_taxi_fares_{len(fares)}.npy", fares)
    return fares


# Generate benchmark files
if __name__ == "__main__":
    load_nyc_taxi_skewed_column(year=2024, month=1, n_rows=1_000_000)
    load_nyc_taxi_skewed_column(year=2024, month=1, n_rows=5_000_000)
    load_nyc_taxi_skewed_column(year=2024, month=1, n_rows=10_000_000)  # or more if you have RAM