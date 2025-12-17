import sys
from pathlib import Path

import pandas as pd


def load_sp500_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S and P features file not found at {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Expected a Date column in sp500_features_daily.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_vix_from_kaggle(path: Path) -> pd.DataFrame:
    """
    Load VIX from the Kaggle bitcoin plus macro dataset.

    Expected columns in df_all_for_kaggle.csv:
    - date
    - vix_close
    plus others we ignore for now.
    """
    if not path.exists():
        raise FileNotFoundError(f"Kaggle macro file not found at {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Expected a date column in df_all_for_kaggle.csv")

    if "vix_close" not in df.columns:
        raise ValueError("Expected a vix_close column in df_all_for_kaggle.csv")

    df["Date"] = pd.to_datetime(df["date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df_vix = df[["Date", "vix_close"]].copy()
    df_vix = df_vix.rename(columns={"vix_close": "vix_level"})

    return df_vix


def merge_sp500_with_vix(
    sp500_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left join S and P daily features with VIX levels on Date.
    Forward fills VIX for missing days inside the sample.
    """
    merged = sp500_df.merge(vix_df, on="Date", how="left")

    merged["vix_level"] = merged["vix_level"].ffill()

    if merged["vix_level"].isna().all():
        raise ValueError(
            "VIX column is all missing after merge. "
            "Check date ranges and column names."
        )

    return merged


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "02_Data" / "Processed"
    external_dir = project_root / "02_Data" / "External"

    sp500_features_path = processed_dir / "sp500_features_daily.csv"
    kaggle_macro_path = external_dir / "df_all_for_kaggle.csv"
    output_path = processed_dir / "sp500_features_daily_with_vix.csv"

    print(f"Loading S and P features from: {sp500_features_path}")
    sp500_df = load_sp500_features(sp500_features_path)

    print(f"Loading Kaggle macro data from: {kaggle_macro_path}")
    vix_df = load_vix_from_kaggle(kaggle_macro_path)

    print("Merging S and P features with VIX")
    merged_df = merge_sp500_with_vix(sp500_df, vix_df)

    print("Merged shape:", merged_df.shape)
    print("Columns:", merged_df.columns.tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"Saved merged features with VIX to: {output_path}")


if __name__ == "__main__":
    main()
