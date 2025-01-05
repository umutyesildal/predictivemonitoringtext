import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, classification_report
import numpy as np
from sklearn.utils import resample
import os
import sys

# Add parent directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Create results directory if it doesn't exist
results_dir = os.path.join(current_dir, "cv_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Set correct data path
data_filepath = os.path.join(parent_dir, "data", "BPI_Challenge_2017.csv")

from PredictiveModel import PredictiveModel

# Load and prepare data
print(f"Loading data from: {data_filepath}")
data = pd.read_csv(data_filepath, sep=";", low_memory=False, dtype=str)

# Data preprocessing
def preprocess_data(df):
    # Convert times to numeric
    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
    df["completeTime"] = pd.to_datetime(df["completeTime"], errors="coerce")
    df["startTime"] = df["startTime"].astype("int64") // 10**9
    df["completeTime"] = df["completeTime"].astype("int64") // 10**9

    # Convert boolean columns
    df["Accepted"] = df["Accepted"].str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    df["Selected"] = df["Selected"].str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    # Convert numeric columns
    numeric_cols = ["RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", 
                    "CreditScore", "NumberOfTerms", "OfferedAmount"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Sort and reindex events
    df.sort_values(by=["case", "startTime"], inplace=True)
    df["event"] = df.groupby("case").cumcount() + 1
    return df

# Define columns and parameters
static_cols = ["RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", "CreditScore"]
dynamic_cols = ["startTime", "completeTime", "EventDescription"]
cat_cols = ["ApplicationType", "LoanGoal", "EventOrigin", "Action"]
case_id_col = "case"
label_col = "Accepted"
event_nr_col = "event"
text_col = "EventDescription"

# Preprocess data
data = preprocess_data(data)

# Detect minority class
minority_class = 0 if (data[label_col] == 0).sum() < (data[label_col] == 1).sum() else 1
majority_class = 1 - minority_class

# Separate majority and minority samples
minority = data[data[label_col] == minority_class]
majority = data[data[label_col] == majority_class]

# Upsample minority class
minority_upsampled = resample(minority,
                              replace=True,
                              n_samples=len(majority),
                              random_state=42)

# Create balanced dataset
data_balanced = pd.concat([majority, minority_upsampled])

# Define parameters for experiments
confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
cls_methods = ["rf"]  # Can add more classifiers if needed

# Base configuration
encoder_kwargs = {
    "event_nr_col": event_nr_col,
    "static_cols": static_cols,
    "dynamic_cols": dynamic_cols,
    "cat_cols": cat_cols,
    "encoding_method": "onehot",
    "oversample_fit": False,
    "minority_label": "0",
    "fillna": True,
    "random_state": 22
}

transformer_kwargs = {
    "ngram_max": 1,
    "alpha": 1.0,
    "nr_selected": 100,
    "pos_label": "1"
}

# Run experiments
results = []
loo = LeaveOneOut()

for train_index, test_index in loo.split(data_balanced):
    train_chunk = data_balanced.iloc[train_index]
    test_chunk = data_balanced.iloc[test_index]

    for cls_method in cls_methods:
        if cls_method == "rf":
            cls_kwargs = {"n_estimators": 100, "random_state": 22}
        else:
            cls_kwargs = {"random_state": 22}

        model = PredictiveModel(
            nr_events=3,
            case_id_col=case_id_col,
            label_col=label_col,
            text_col=text_col,
            text_transformer_type=None,
            cls_method=cls_method,
            encoder_kwargs=encoder_kwargs,
            transformer_kwargs=transformer_kwargs,
            cls_kwargs=cls_kwargs
        )

        model.fit(train_chunk)

        # Avoid processing empty test sets
        if test_chunk.empty:
            continue

        preds_proba = model.predict_proba(test_chunk)
        y_true = test_chunk[label_col].values

        for conf in confidences:
            mask = (preds_proba.max(axis=1) >= conf)
            if mask.any():
                y_pred = preds_proba[mask].argmax(axis=1)
                y_true_filtered = y_true[mask]

                precision = precision_score(y_true_filtered, y_pred, zero_division=0)
                recall = recall_score(y_true_filtered, y_pred, zero_division=0)
                acc = accuracy_score(y_true_filtered, y_pred)
                auc = roc_auc_score(y_true_filtered, preds_proba[mask][:, 1]) if len(np.unique(y_true_filtered)) > 1 else None

                results.append({
                    'confidence': conf,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': acc,
                    'auc': auc
                })

# Aggregate results
results_df = pd.DataFrame(results)
print("\nLOOCV Metrics:")
print(f"Average Precision: {results_df['precision'].mean()}")
print(f"Average Recall: {results_df['recall'].mean()}")
print(f"Average Accuracy: {results_df['accuracy'].mean()}")
print(f"Average AUC: {results_df['auc'].mean()}")
