import os
import json
import pandas as pd
from collections import OrderedDict
from datasets import Dataset, DatasetDict

# Paths to data
metadata_file = "./data/norec_translated/metadata_v_2_0.json"
train_folder = "./data/norec_translated/train"
# Uncomment these when dev and test data are available
# dev_folder = "./data/norec_translated/dev"
# test_folder = "./data/norec_translated/test"


def convert_rating(rating):
    """
    Converts a numeric rating into a 3-class sentiment label.
    Mapping: ratings 1-2 -> 0 (Negative), 3-4 -> 1 (Neutral), 5-6 -> 2 (Positive)
    Adjust thresholds as needed.
    """
    if rating < 3:
        return 0
    elif rating < 5:
        return 1
    else:
        return 2


# Load metadata from JSON file in the exact order it appears
with open(metadata_file, "r", encoding="utf-8") as f:
    metadata = json.loads(f.read(), object_pairs_hook=OrderedDict)

data = []
missing_files = []

# Process entries for the train split in the original JSON order
for key, info in metadata.items():
    if info.get("split") != "train":
        continue

    file_path = os.path.join(train_folder, f"{key}.txt")

    # Attempt to read the corresponding text file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        # If the file is missing, store the name and skip
        missing_files.append(file_path)
        continue
    except Exception as e:
        # Log any other file-read issue and skip
        print(f"Error reading {file_path}: {e}")
        missing_files.append(file_path)
        continue

    rating = info.get("rating")
    if rating is None:
        # If rating is missing, you could handle or skip similarly
        print(f"Missing rating for {key}. Skipping.")
        continue

    label = convert_rating(rating)
    data.append({"text": text, "label": label})

df = pd.DataFrame(data)
print(f"Loaded {len(df)} samples from the train split.")

# Create the Hugging Face DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(df),
    # When dev and test splits are ready, do similarly for dev/test:
    # "validation": Dataset.from_pandas(dev_df),
    # "test": Dataset.from_pandas(test_df),
})

dataset.save_to_disk("./data_proc/processed_norec_translated_dataset")
print("✅ Norec Translated dataset processed and saved!")

# Final summary of missing files
if missing_files:
    print(
        f"Processing finished. {len(missing_files)} files were not processed:")
    for mf in missing_files:
        print(f" - {mf}")
else:
    print("Processing finished. All files were processed successfully.")
