import pandas as pd
import re
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Path to the Trustpilot dataset file
trustpilot_file = "./data/trustpilot/trustpilot_reviews.tsv"

def load_and_clean_trustpilot_data(file_path):
    """Loads, cleans, and processes the Trustpilot dataset."""
    data = []
    skipped_lines = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                skipped_lines += 1
                continue

            # Remove excessive whitespace
            line = re.sub(r"\s+", " ", line).strip()

            # Regex to capture text and sentiment at the end
            match = re.match(r"(.+?)\s*(-?\d+)$", line)  # Capture text + last number as label

            if match:
                text, sentiment = match.groups()

                try:
                    label = int(sentiment)  # Convert label to integer

                    # Convert 5-point scale to 3-class sentiment
                    if label <= 2:
                        label = 0  # Negative Sentiment
                    elif label == 3:
                        label = 1  # Neutral Sentiment
                    elif label >= 4:
                        label = 2  # Positive Sentiment
                    else:
                        raise ValueError("Invalid label")  # Catch unexpected values

                    # Append cleaned data
                    data.append({"text": text, "label": label})

                except ValueError:
                    skipped_lines += 1
            else:
                skipped_lines += 1

    print(f"✅ {file_path}: Loaded {len(data)} rows, Skipped {skipped_lines} malformed rows.")
    return pd.DataFrame(data)

# Load and preprocess Trustpilot dataset
df = load_and_clean_trustpilot_data(trustpilot_file)

# Ensure the test set has an equal number of samples per class
num_samples_per_class = min(df["label"].value_counts().min(), 500)  # Ensure balance

test_df_balanced = (
    df.groupby("label")
    .apply(lambda x: x.sample(n=num_samples_per_class, random_state=42, replace=False))
    .reset_index(drop=True)
)

# Remove test set examples from the remaining dataset
remaining_df = df.drop(test_df_balanced.index)

# Splitting remaining data into train (80%) and validation (10%)
train_data, val_data = train_test_split(
    remaining_df, test_size=0.2, stratify=remaining_df["label"], random_state=42
)

# Convert to Hugging Face Dataset format
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "validation": Dataset.from_pandas(val_data),
    "test": Dataset.from_pandas(test_df_balanced),
})

# Save dataset
dataset.save_to_disk("./processed_trustpilot_dataset")

print(f"✅ Trustpilot dataset processed and saved successfully!")
print(f"✅ Test Set Balanced: {test_df_balanced['label'].value_counts().to_dict()} (equal number per class)")
print(f"✅ Train Set Size: {len(train_data)}, Validation Set Size: {len(val_data)}, Test Set Size: {len(test_df_balanced)}")
