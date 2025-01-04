import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys
import os

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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

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
dynamic_cols = ["startTime", "completeTime"]
cat_cols = ["ApplicationType", "LoanGoal", "EventOrigin", "Action"]
case_id_col = "case"
label_col = "Accepted"
event_nr_col = "event"

# Preprocess data
data = preprocess_data(data)

# Divide into train and test data
train_names, test_names = train_test_split(data[case_id_col].unique(), train_size=0.8, random_state=42)
train = data[data[case_id_col].isin(train_names)]
test = data[data[case_id_col].isin(test_names)]

# Create five folds for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

# Define parameters for experiments
confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
cls_methods = ["rf"]  # Can add more classifiers like "logit" if needed

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
part = 1
for train_index, test_index in kf.split(train[train[event_nr_col]==1], train[train[event_nr_col]==1][label_col]):
    
    # Create train and validation data for current fold
    current_train_names = train_names[train_index]
    train_chunk = train[train[case_id_col].isin(current_train_names)]
    current_test_names = train_names[test_index]
    test_chunk = train[train[case_id_col].isin(current_test_names)]
    
    for cls_method in cls_methods:
        # Define classifier parameters
        if cls_method == "rf":
            cls_kwargs = {"n_estimators": 100, "random_state": 22}
        else:
            cls_kwargs = {"random_state": 22}
        
        # Initialize and train model
        model = PredictiveModel(
            nr_events=3,
            case_id_col=case_id_col,
            label_col=label_col,
            text_col="text",
            text_transformer_type=None,
            cls_method=cls_method,
            encoder_kwargs=encoder_kwargs,
            transformer_kwargs=transformer_kwargs,
            cls_kwargs=cls_kwargs
        )
        
        # Train model
        model.fit(train_chunk)
        
        # Test and evaluate
        preds_proba = model.predict_proba(test_chunk)
        y_true = model.test_y
        
        # Calculate metrics for different confidence thresholds
        results = []
        for conf in confidences:
            # Apply confidence threshold
            mask = (preds_proba.max(axis=1) >= conf)
            if mask.any():
                y_pred = preds_proba[mask].argmax(axis=1)
                y_true_filtered = y_true[mask]
                
                # Calculate metrics
                acc = accuracy_score(y_true_filtered, y_pred)
                auc = roc_auc_score(y_true_filtered, preds_proba[mask][:,1])
                coverage = len(y_pred) / len(y_true)
                
                results.append({
                    'confidence': conf,
                    'accuracy': acc,
                    'auc': auc,
                    'coverage': coverage,
                    'fold': part
                })
        
        # Save results with full path
        results_df = pd.DataFrame(results)
        result_filepath = os.path.join(results_dir, f'ohe_{cls_method}_part{part}.csv')
        results_df.to_csv(result_filepath, index=False)
        print(f"Saved results to: {result_filepath}")
        
        # Print fold results
        print(f"\nFold {part} Results for {cls_method}:")
        print(results_df)
    
    part += 1

# Final evaluation on test set
print("\nFinal Evaluation on Test Set:")
final_model = PredictiveModel(
    nr_events=3,
    case_id_col=case_id_col,
    label_col=label_col,
    text_col="text",
    text_transformer_type=None,
    cls_method="rf",
    encoder_kwargs=encoder_kwargs,
    transformer_kwargs=transformer_kwargs,
    cls_kwargs={"n_estimators": 100, "random_state": 22}
)

final_model.fit(train)
final_preds_proba = final_model.predict_proba(test)
final_preds = final_preds_proba.argmax(axis=1)

print("\nTest Set Metrics:")
print("Accuracy:", accuracy_score(final_model.test_y, final_preds))
print("AUC:", roc_auc_score(final_model.test_y, final_preds_proba[:,1]))
print("\nClassification Report:")
print(classification_report(final_model.test_y, final_preds))
