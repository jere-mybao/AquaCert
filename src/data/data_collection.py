import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_params(filepath: str) -> float:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")


def load_data_from_url(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url, sep=",")
    except Exception as e:
        raise Exception(f"Error loading data from {url}: {e}")


def split_data(
    data: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")


def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")


def main():
    data_url = "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/refs/heads/main/water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw")

    try:
        # Load data from URL
        data = load_data_from_url(data_url)

        # Load parameters
        test_size = load_params(params_filepath)

        # Split data
        train_data, test_data = split_data(data, test_size)

        # Create directories if not exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save train and test data
        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()