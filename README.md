# Sentiment Datasets Processing and Structure
This document provides an overview of how the datasets have been processed, the sentiment labels assigned, and the structure of the dataset splits (train, validation, test).

The data has been processed and can be found in [data_proc](./data_proc). For information about sources and processing methods, please see below.

## Overview of Dataset Processing

### Swedish ABSAbank-Imm Dataset
**Source**: [Språkbanken](https://spraakbanken.gu.se/en/resources/absabank-imm)

**Format**: TSV (tab-separated values)

**Processing Steps**:
1. Loaded dataset from .tsv files.
2. Ensured required columns (text, label) exist.
3. Rounded labels and converted a 5-point sentiment scale to a 3-class system:
   * 0-2 → Negative (0)
   * 3 → Neutral (1)
   * 4-5 → Positive (2)
4. Split into train (80%), validation (10%), and test (10%).

**Final Split**:
* Train: 3,860
* Validation: 486
* Test: 484

### Trustpilot Reviews Dataset

**Source**: scraped using this [webscraper](https://github.com/TeaElming/trustpilot-scraper) *(Scraped on 19/02/2025)*

**Format**: TSV (tab-separated values)

**Processing Steps**:
1. Read dataset and handled cases with multiple tab-separated fields.
2. Extracted text and assigned the last tab-separated value as sentiment label.
3. Converted a 5-point sentiment scale to a 3-class system:
     * 0-2 → Negative (0)
     * 3 → Neutral (1)
     * 4-5 → Positive (2)
4. Ensured the test set is balanced, containing an equal number of Negative, Neutral, and Positive samples.
5. Split remaining data into train (80%) and validation (10%).

**Final Split**:
* Train: 15,829
* Validation: 3,958
* Test: 1,500 (Balanced: 500 Negative, 500 Neutral, 500 Positive)

### Niklas-Palm-Twitter Dataset
**Source**: [Niklas Palm GitHub](https://github.com/niklas-palm/sentiment-classification/tree/master/data)

**Format**: CSV

**Processing Steps**:
1. Read CSV files and removed excessive whitespace.
2. Extracted text and last numeric value as sentiment label.
3. Converted sentiment labels from (-1, 0, 1) to (0, 1, 2):
   * -1 → Negative (0)
   * 0 → Neutral (1)
   * 1 → Positive (2)
4. Balanced the test set to ensure equal representation of all three sentiment classes.
5. Split remaining data into train (80%) and validation (10%).

**Final Split**:
* Train: 8,465
* Validation: 2,117
* Test: 1,500 (Balanced: 500 Negative, 500 Neutral, 500 Positive)

## Sentiment Labels & Meaning
| Label | Meaning  |
|-------|----------|
| 0     | Negative |
| 1     | Neutral  |
| 2     | Positive |

This labeling ensures consistency across all datasets.

## Dataset Structure:
Train / Validation / Test
* Train Set (80%)
  * Used to train machine learning models.
  * Largest portion of the dataset.
* Validation Set (10%)
  * Used for hyperparameter tuning and model evaluation before testing.
* Test Set (10%)
  * Held-out dataset to evaluate the model's final performance.
  * Balanced for Trustpilot and Twitter (equal negative, neutral, positive). ABSAbank provided a test set initially, which has been kept as it was.


## How to Use These Datasets
* For Model Training: Use the train set for learning patterns in text data.
* For Hyperparameter Tuning: Use the validation set to adjust model parameters.
* For Final Evaluation: Use the test set to assess how well the model generalises.
