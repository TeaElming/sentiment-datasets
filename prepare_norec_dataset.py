"""
Create balanced train / validation / test splits for the
NOREC-translated corpus and save them as a Hugging-Face DatasetDict.

Output folder: ./processed_norec_translated_dataset
(removed and recreated on every run)
"""

import os
import json
import shutil
import pandas as pd
from collections import OrderedDict
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import tqdm

# ──────────────────────────────────────────────────────────── paths
ROOT_DIR   = "./data/norec_translated"
TXT_DIR  = os.path.join(ROOT_DIR, "train")
META_FILE  = os.path.join(ROOT_DIR, "metadata_v_2_0.json")
OUT_DIR    = "./processed_norec_translated_dataset"

# ──────────────────────────────────────────────────── helpers / load
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm  # pip install tqdm   ▸ nice progress bar (optional)

def convert_rating(r: int) -> int:
    return 0 if r < 3 else (1 if r < 5 else 2)

def _read_one(item):
    """Worker: read a single TXT file; return row-dict or None."""
    doc_id, info = item
    path = os.path.join(TXT_DIR, f"{doc_id}.txt")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        return {"text": text, "label": convert_rating(info["rating"])}
    except FileNotFoundError:
        return None

def load_reviews(workers: int | None = None) -> pd.DataFrame:
    """Read all reviews in parallel; returns DataFrame."""
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f, object_pairs_hook=OrderedDict)

    rows, missing = [], 0
    workers = workers or os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # tqdm is optional – remove if you don’t want the bar
        for row in tqdm.tqdm(pool.map(_read_one, meta.items()), total=len(meta)):
            if row is None:
                missing += 1
            else:
                rows.append(row)

    if missing:
        print(f"⚠️  {missing} TXT files listed in metadata were missing.")
    print(f"✅ Loaded {len(rows)} usable reviews "
          f"(threads used: {workers}).")

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────── splits
def make_splits(df: pd.DataFrame):
    # balanced test set (≤ 500 per class or the class minimum)
    per_cls = min(df["label"].value_counts().min(), 500)
    test_df = (
        df.groupby("label")
          .apply(lambda x: x.sample(n=per_cls, random_state=42, replace=False))
          .reset_index(drop=True)
    )
    remaining_df = df.drop(test_df.index)
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=0.2,
        stratify=remaining_df["label"],
        random_state=42,
    )
    return train_df, val_df, test_df

# ────────────────────────────────────────────────────── save dataset
def save_dataset(train_df, val_df, test_df):
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    ds = DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    ds.save_to_disk(OUT_DIR)

    print(
        f"✅ Saved to {OUT_DIR} │ "
        f"train {len(train_df)} │ val {len(val_df)} │ "
        f"test {len(test_df)} (each class ≤ {test_df['label'].value_counts().max()})"
    )

# ────────────────────────────────────────────────────────── main
if __name__ == "__main__":
    df = load_reviews()
    train_df, val_df, test_df = make_splits(df)
    save_dataset(train_df, val_df, test_df)
