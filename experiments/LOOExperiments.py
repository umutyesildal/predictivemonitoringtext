import pandas as pd
from sklearn.model_selection import LeaveOneOut
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.utils import resample

# Add parent directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Create results directory if it doesn't exist
results_dir = os.path.join(current_dir, "loo_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Set correct data path
data_filepath = os.path.join(parent_dir, "data", "BPI_Challenge_2017.csv")

from PredictiveModel import PredictiveModel

# Load and prepare data
print(f"Loading data from: {data_filepath}")
data = pd.read_csv(data_filepath, sep=";", low_memory=False, dtype=str)

def preprocess_data(df):
    print("Initial shape:", df.shape)
    print("Initial columns:", df.columns.tolist())
    
    # First, get last event for each case
    df = df.sort_values(['case', 'startTime']).groupby('case').last().reset_index()
    
    # Then do all the conversions
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

    # Add event number (should be 1 for all since we took last events)
    df["event"] = 1
    
    return df

# Define columns and parameters
static_cols = ["RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", "CreditScore"]
dynamic_cols = ["startTime", "completeTime"]
cat_cols = ["ApplicationType", "LoanGoal", "EventOrigin", "Action"]
case_id_col = "case"
label_col = "Accepted"
event_nr_col = "event"

# Load and preprocess data
print(f"Loading data from: {data_filepath}")
data = pd.read_csv(data_filepath, sep=";", low_memory=False, dtype=str)
data = preprocess_data(data)

# Print column information for debugging
print("\nAvailable columns after preprocessing:", data.columns.tolist())
print("\nStatic columns check:")
for col in static_cols:
    print(f"{col}: {col in data.columns}")
    if col in data.columns:
        print(f"Sample values: {data[col].head()}")

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

print("\nFinal balanced shape:", data_balanced.shape)
print("Final balanced class distribution:")
print(data_balanced[label_col].value_counts())
print("\nStatic columns in final balanced data:")
print(data_balanced[static_cols].head())

# Verify data is not empty
if data_balanced.empty:
    raise ValueError("Dataset is empty after preprocessing!")

if len(data_balanced) < 2:
    raise ValueError(f"Not enough samples for LOO CV. Found only {len(data_balanced)} samples.")

# Initialize Leave-One-Out cross validator
loo = LeaveOneOut()

# Store results
all_predictions = []
all_true_values = []
confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Base configuration
encoder_kwargs = {
    "event_nr_col": event_nr_col,
    "static_cols": static_cols,
    "dynamic_cols": dynamic_cols,
    "cat_cols": cat_cols,
    "encoding_method": "onehot",
    "oversample_fit": True,
    "fillna": True,
    "random_state": 22
}

transformer_kwargs = {
    "ngram_max": 1,
    "alpha": 1.0,
    "nr_selected": 100,
    "pos_label": "1"
}

# Run LOO CV
print("\nStarting Leave-One-Out Cross Validation...")
for train_idx, test_idx in loo.split(data_balanced):
    # Split data
    train_data = data_balanced.iloc[train_idx]
    test_data = data_balanced.iloc[test_idx]
    
    # Initialize and train model
    model = PredictiveModel(
        nr_events=1,
        case_id_col=case_id_col,
        label_col=label_col,
        text_col=None,
        text_transformer_type=None,
        cls_method="rf",
        encoder_kwargs=encoder_kwargs,
        transformer_kwargs=transformer_kwargs,
        cls_kwargs={"n_estimators": 100, "random_state": 22}
    )
    
    # Train and predict
    model.fit(train_data)
    pred_proba = model.predict_proba(test_data)
    pred = pred_proba.argmax(axis=1)
    
    # Store results
    all_predictions.append(pred[0])
    all_true_values.append(model.test_y.iloc[0])

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_true_values = np.array(all_true_values)

# Calculate and print final metrics
print("\nFinal LOO Cross-Validation Results:")
print("Accuracy:", accuracy_score(all_true_values, all_predictions))
try:
    if len(np.unique(all_true_values)) > 1:
        auc = roc_auc_score(all_true_values, all_predictions)
        print("AUC:", auc)
    else:
        print("AUC: Not calculable (only one class present)")
except IndexError:
    print("AUC: Not calculable (prediction error)")

print("\nClassification Report:")
print(classification_report(all_true_values, all_predictions))

# Save results
results_df = pd.DataFrame({
    'true_values': all_true_values,
    'predictions': all_predictions
})
results_df.to_csv(os.path.join(results_dir, 'loo_results.csv'), index=False)
print(f"\nResults saved to: {os.path.join(results_dir, 'loo_results.csv')}")
