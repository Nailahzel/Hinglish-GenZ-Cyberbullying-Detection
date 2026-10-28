import pandas as pd
from datasets import Dataset, concatenate_datasets
import os

def load_all(raw_dir):
    parts = []
    for name in os.listdir(raw_dir):
        if name.endswith(".csv"):
            path = os.path.join(raw_dir, name)
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            if "text" not in df.columns or "label" not in df.columns:
                print(f"⚠️ Skipping {name}: Missing required columns")
                continue

            df = df[["text", "label"]].dropna()
            df["text"] = df["text"].astype(str)
            df["label"] = df["label"].astype(int)

            parts.append(Dataset.from_pandas(df))

    if not parts:
        raise ValueError("❌ No valid datasets found in data/raw")

    # Merge all datasets
    data = concatenate_datasets(parts)

    # Shuffle + Split
    return data.shuffle(seed=42).train_test_split(test_size=0.2)
