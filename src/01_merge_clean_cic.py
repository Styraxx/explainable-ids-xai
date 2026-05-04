import os, glob
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw", "cic_ids2017")
OUT_PARQUET = os.path.join("data", "processed", "cic_ids2017_clean.parquet")
SUMMARY_OUT = os.path.join("outputs", "tables", "cic_dataset_summary.csv")

DROP_COLS = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64":
            df[c] = df[c].astype("int32")
    return df

def main():
    csv_files = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {RAW_DIR}")

    print(f"Found {len(csv_files)} CSV files under {RAW_DIR}")

    frames = []
    total_rows = 0

    for path in csv_files:
        print(f"\nReading: {os.path.basename(path)}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            raise ValueError(f"No 'Label' column found in file: {path}")

        # Drop non-features early if present
        drop = [c for c in df.columns if c in DROP_COLS]
        if drop:
            df = df.drop(columns=drop)

        # Preserve original label for future multi-class
        df["label_original"] = df[label_col].astype(str).str.strip()

        # Binary label (BENIGN=0, ATTACK=1)
        df["label_binary"] = (df["label_original"].str.upper() != "BENIGN").astype("int8")

        # Drop original label column if it's separate
        if label_col != "label_original":
            df = df.drop(columns=[label_col])

        # Remove inf and NaN rows (common in CIC-IDS2017)
        df = df.replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna()
        after = len(df)

        df = reduce_memory(df)

        total_rows += after
        print(f"Rows kept: {after:,} (dropped {before-after:,})")

        frames.append(df)

    print("\nConcatenating all files...")
    full = pd.concat(frames, ignore_index=True)
    print(f"Final merged shape: {full.shape}")

    # Summary for thesis/supporting materials
    summary = pd.DataFrame({
        "rows": [len(full)],
        "columns": [len(full.columns)],
        "benign_count": [int((full["label_binary"] == 0).sum())],
        "attack_count": [int((full["label_binary"] == 1).sum())],
        "attack_rate": [float((full["label_binary"] == 1).mean())],
    })

    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False)

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    full.to_parquet(OUT_PARQUET, index=False)

    print(f"\nSaved cleaned dataset → {OUT_PARQUET}")
    print(f"Saved dataset summary  → {SUMMARY_OUT}")

if __name__ == "__main__":
    main()
