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

# Add sampling parameter at the top
MAX_SAMPLES = 2000  # Adjust this number based on your RAM

def preprocess_data(df):
    print("Initial shape:", df.shape)
    
    # First, get last event for each case
    df = df.sort_values(['case', 'startTime']).groupby('case').last().reset_index()
    
    # Sample if dataset is too large
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
    
    # Conversions
    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
    df["completeTime"] = pd.to_datetime(df["completeTime"], errors="coerce")
    df["startTime"] = df["startTime"].astype("int64") // 10**9
    df["completeTime"] = df["completeTime"].astype("int64") // 10**9
    df["Accepted"] = df["Accepted"].str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    df["Selected"] = df["Selected"].str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    # Convert and normalize numeric columns
    numeric_cols = ["RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", 
                   "CreditScore", "NumberOfTerms", "OfferedAmount"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        # Normalize numeric columns
        if df[col].std() != 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Advanced feature engineering
    df['process_duration'] = df['completeTime'] - df['startTime']
    df['process_duration'] = (df['process_duration'] - df['process_duration'].mean()) / df['process_duration'].std()
    
    # More meaningful financial ratios
    df['monthly_burden'] = df['MonthlyCost'] * df['NumberOfTerms'] / df['RequestedAmount']
    df['credit_ratio'] = df['CreditScore'] / 1000  # Normalize credit score
    df['offer_ratio'] = df['OfferedAmount'] / df['RequestedAmount']
    
    # Replace infinities and fill NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    df["event"] = 1
    return df

# Define columns and parameters
static_cols = [
    "RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", "CreditScore",
    "monthly_burden", "credit_ratio", "offer_ratio", "process_duration"
]
dynamic_cols = ["startTime", "completeTime", "process_duration"]
cat_cols = ["ApplicationType", "LoanGoal"]  # Reduced to most important ones
case_id_col = "case"
label_col = "Accepted"
event_nr_col = "event"

# Load and preprocess data with sampling
print(f"Loading data from: {data_filepath}")
data = pd.read_csv(data_filepath, sep=";", low_memory=False, dtype=str)
data = preprocess_data(data)

# Balance classes with smaller numbers
min_class_size = min(len(data[data[label_col] == 0]), len(data[data[label_col] == 1]))
samples_per_class = min(min_class_size, MAX_SAMPLES // 2)  # Ensure total samples don't exceed MAX_SAMPLES

# Sınıf dengeleme stratejisini güncelleyelim
def balance_dataset(df, label_col, n_samples=None):
    if n_samples is None:
        n_samples = min(len(df[df[label_col] == 0]), len(df[df[label_col] == 1]))
    
    # Stratified sampling yapalım
    class_0 = df[df[label_col] == 0].sample(n=n_samples, random_state=42)
    class_1 = df[df[label_col] == 1].sample(n=n_samples, random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([class_0, class_1])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Veri setini dengele
data_balanced = balance_dataset(data, label_col, samples_per_class)

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

# Optimize Random Forest parameters
cls_kwargs = {
    "n_estimators": 500,  # Increased
    "max_depth": 15,      # Increased
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "class_weight": "balanced",  # Added class weights
    "random_state": 22
}

# Run LOO CV with less output
print("\nStarting Leave-One-Out Cross Validation...")
total_iterations = len(data_balanced)
for idx, (train_idx, test_idx) in enumerate(loo.split(data_balanced)):
    if idx % 100 == 0:  # Reduced progress updates
        print(f"Progress: {idx}/{total_iterations}")
    
    train_data = data_balanced.iloc[train_idx]
    test_data = data_balanced.iloc[test_idx]
    
    model = PredictiveModel(
        nr_events=1,
        case_id_col=case_id_col,
        label_col=label_col,
        text_col=None,
        text_transformer_type=None,
        cls_method="rf",
        encoder_kwargs=encoder_kwargs,
        transformer_kwargs=transformer_kwargs,
        cls_kwargs=cls_kwargs
    )
    
    # Suppress output during training
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_data)
        pred_proba = model.predict_proba(test_data)
    
    pred = pred_proba.argmax(axis=1)
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
