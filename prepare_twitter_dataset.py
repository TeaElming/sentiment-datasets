import pandas as pd
import re
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Paths to dataset files
train_file = "./data/niklas-palm-twitter/train_set.csv"
test_file = "./data/niklas-palm-twitter/balanced_test_set.csv"


def load_and_clean_data(file_path):
    """Loads, cleans, and processes the dataset into the desired format."""
    data = []
    skipped_lines = 0

    with open(file_path, "r", encoding="utf-8") as file:
        next(file)  # Skip header
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                skipped_lines += 1
                continue

            # Remove excessive whitespace
            line = re.sub(r"\s+", " ", line).strip()

            # Regex to capture text and sentiment at the end
            match = re.match(r"(.+?),(-1|0|1)$", line)

            if match:
                text, sentiment = match.groups()

                # Convert labels: -1 → 0 (Negative), 0 → 1 (Neutral), 1 → 2 (Positive)
                label = int(sentiment) + 1

                # Append cleaned data
                data.append({"text": text, "label": label})
            else:
                skipped_lines += 1

    print(
        f"✅ {file_path}: Loaded {len(data)} rows, Skipped {skipped_lines} malformed rows.")
    return pd.DataFrame(data)


# Load datasets
train_df = load_and_clean_data(train_file)
test_df = load_and_clean_data(test_file)

# Combine both datasets
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Ensure all three sentiment classes have equal distribution in the test set
# Ensuring balance while limiting to 500 per class
num_samples_per_class = min(combined_df["label"].value_counts().min(), 500)

test_df_balanced = (
    combined_df.groupby("label")
    .apply(lambda x: x.sample(n=num_samples_per_class, random_state=42, replace=False))
    .reset_index(drop=True)
)

# Remove test set examples from the remaining dataset
remaining_df = combined_df.drop(test_df_balanced.index)

# Splitting remaining data into train (80%) and validation (10%)
train_data, dev_data = train_test_split(
    remaining_df, test_size=0.2, stratify=remaining_df["label"], random_state=42
)

# Convert to Hugging Face Dataset format
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "validation": Dataset.from_pandas(dev_data),
    "test": Dataset.from_pandas(test_df_balanced),
})

# Save dataset
dataset.save_to_disk("./processed_twitter_dataset")

print(f"✅ Niklas-Palm-Twitter dataset processed and saved successfully!")
print(
    f"✅ Test Set Balanced: {test_df_balanced['label'].value_counts().to_dict()} (equal number per class)")
print(
    f"✅ Train Set Size: {len(train_data)}, Validation Set Size: {len(dev_data)}, Test Set Size: {len(test_df_balanced)}")
