import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

# Paths to dataset files
train_file = "./data/absabank-imm/absabank-imm_train.tsv"
dev_file = "./data/absabank-imm/absabank-imm_dev.tsv"
test_file = "./data/absabank-imm/absabank-imm_test.tsv"


def load_data(file_path):
    """Loads the dataset from TSV and preprocesses labels"""
    df = pd.read_csv(file_path, sep="\t")

    # Ensure we have the correct columns
    assert "text" in df.columns and "label" in df.columns, "Missing required columns!"

    # Convert labels to numeric and drop missing labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    # ✅ Round labels to nearest integer before classification
    df["label"] = df["label"].round().astype(int)

    # ✅ Convert 5-point scale to 3-class sentiment (0: Negative, 1: Neutral, 2: Positive)
    def convert_labels(label):
        if label <= 2:  # Negative Sentiment
            return 0
        elif label == 3:  # Neutral Sentiment
            return 1
        elif label >= 4:  # Positive Sentiment
            return 2
        else:
            return np.nan  # Handle unexpected values

    df["label"] = df["label"].apply(convert_labels)

    return df


# Load train, dev, and test sets
train_df = load_data(train_file)
dev_df = load_data(dev_file)
test_df = load_data(test_file)

# Convert to Hugging Face Dataset format
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(dev_df),
    "test": Dataset.from_pandas(test_df),
})

# Save dataset for later use
dataset.save_to_disk("./processed_absabank_dataset")

print("✅ Swedish ABSAbank-Imm dataset processed and saved!")
