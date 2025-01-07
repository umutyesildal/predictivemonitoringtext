import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys
import os
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

# Add parent directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Create results directory if it doesn't exist
results_dir = os.path.join(current_dir, "cv_results")
final_results_dir = os.path.join(current_dir, "final_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")
if not os.path.exists(final_results_dir):
    os.makedirs(final_results_dir)
    print(f"Created final results directory: {final_results_dir}")

# Set correct data path
data_filepath = os.path.join(parent_dir, "data", "BPI_Challenge_2017.csv")

from PredictiveModel import PredictiveModel
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Load and prepare data
print(f"Loading data from: {data_filepath}")
data = pd.read_csv(data_filepath, sep=";", low_memory=False, dtype=str)

# Data preprocessing
def preprocess_data(df):
    print("\nPreprocessing Data...")
    
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
static_cols = [
    "RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", 
    "CreditScore", "NumberOfTerms", "OfferedAmount"
]
dynamic_cols = ["startTime", "completeTime"]
cat_cols = ["ApplicationType", "LoanGoal", "EventOrigin"]
case_id_col = "case"
label_col = "Accepted"
event_nr_col = "event"
text_col = "EventDescription"
event_col = "Action"

# Print data info before preprocessing
print("\nData Info Before Preprocessing:")
print(data.info())

# Preprocess data
data = preprocess_data(data)

# Print data info after preprocessing
print("\nData Info After Preprocessing:")
print(data.info())

# Print key statistics
print("\nData Statistics:")
print(f"Total cases: {data[case_id_col].nunique():,}")
print(f"Total events: {len(data):,}")
print(f"Average events per case: {len(data) / data[case_id_col].nunique():.2f}")
print(f"Label distribution:\n{data.groupby(case_id_col)[label_col].first().value_counts(normalize=True)}")

# First create case-level labels
case_labels = data.groupby(case_id_col)[label_col].first()

# Then perform train-test split with larger test size for better evaluation
train_names, test_names = train_test_split(case_labels.index.values, 
                                         test_size=0.3,
                                         random_state=42,
                                         stratify=case_labels.values)

# Split into train and test sets
train = data[data[case_id_col].isin(train_names)]
test = data[data[case_id_col].isin(test_names)]

print("\nSplit Statistics:")
print(f"Training set cases: {len(train_names)}")
print(f"Test set cases: {len(test_names)}")

# Get train case labels
train_case_labels = case_labels[case_labels.index.isin(train_names)]

# Create five folds for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

# Convert train_names to array for proper indexing
train_names_array = np.array(train_names)

# Define parameters for experiments
confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
cls_methods = ["rf"]  # Can add more classifiers like "logit" if needed

# Base configuration
encoder_kwargs = {
    "event_nr_col": event_nr_col,
    "static_cols": static_cols,
    "dynamic_cols": dynamic_cols,
    "cat_cols": cat_cols,
    "event_col": event_col,
    "encoding_method": "onehot",
    "oversample_fit": False,
    "minority_label": "0",
    "fillna": True,
    "random_state": 22
}

# Text transformer parameters for BoNGTransformer
transformer_kwargs = {
    "ngram_min": 1,
    "ngram_max": 1,
    "tfidf": False,
    "nr_selected": 100
}

# Run experiments
part = 1
for train_index, test_index in kf.split(train_names_array, train_case_labels):
    print(f"\nProcessing Fold {part}...")
    
    # Create train and validation data for current fold
    current_train_names = train_names_array[train_index]
    train_chunk = train[train[case_id_col].isin(current_train_names)]
    current_test_names = train_names_array[test_index]
    test_chunk = train[train[case_id_col].isin(current_test_names)]
    
    for cls_method in cls_methods:
        # Initialize and train model
        model = PredictiveModel(
            nr_events=5,
            case_id_col=case_id_col,
            label_col=label_col,
            text_col=text_col,
            text_transformer_type="bong",
            cls_method=cls_method,
            encoder_kwargs=encoder_kwargs,
            transformer_kwargs=transformer_kwargs,
            cls_kwargs={
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 22
            }
        )
        
        # Train model
        model.fit(train_chunk)

        # Test and evaluate
        preds_proba = model.predict_proba(test_chunk)
        y_true = model.test_y
        
        # Calculate metrics for different confidence thresholds
        results = []
        for conf in confidences:
            mask = (preds_proba.max(axis=1) >= conf)
            if mask.any():
                y_pred = preds_proba[mask].argmax(axis=1)
                y_true_filtered = y_true[mask]
                
                acc = accuracy_score(y_true_filtered, y_pred)
                
                try:
                    if len(np.unique(y_true_filtered)) > 1:
                        auc = roc_auc_score(y_true_filtered, preds_proba[mask][:,1])
                    else:
                        auc = None
                except IndexError:
                    auc = None
                    
                coverage = len(y_pred) / len(y_true)
                
                results.append({
                    'confidence': conf,
                    'accuracy': acc,
                    'auc': auc,
                    'coverage': coverage,
                    'fold': part
                })
        
        results_df = pd.DataFrame(results)
        result_filepath = os.path.join(results_dir, f'ohe_{cls_method}_part{part}.csv')
        results_df.to_csv(result_filepath, index=False)
        
        print(f"\nFold {part} Results:")
        print(results_df)
    
    part += 1

# Final evaluation
print("\nFinal Evaluation on Test Set:")
final_model = PredictiveModel(
    nr_events=5,
    case_id_col=case_id_col,
    label_col=label_col,
    text_col=text_col,
    text_transformer_type="bong",
    cls_method="rf",
    encoder_kwargs=encoder_kwargs,
    transformer_kwargs=transformer_kwargs,
    cls_kwargs={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 22
    }
)

final_model.fit(train)
final_preds_proba = final_model.predict_proba(test)
final_preds = final_preds_proba.argmax(axis=1)

print("\nTest Set Metrics:")
print("Accuracy:", accuracy_score(final_model.test_y, final_preds))
try:
    if len(np.unique(final_model.test_y)) > 1:
        auc = roc_auc_score(final_model.test_y, final_preds_proba[:,1])
        print("AUC:", auc)
except IndexError:
    print("AUC: Not calculable")
print("\nClassification Report:")
print(classification_report(final_model.test_y, final_preds))

# After final evaluation, calculate and save detailed metrics
print("\nCalculating and saving detailed metrics...")
confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
detailed_results = []

for conf in confidences:
    mask = (final_preds_proba.max(axis=1) >= conf)
    if mask.any():
        y_pred = final_preds_proba[mask].argmax(axis=1)
        y_true_filtered = final_model.test_y[mask]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate failure rate (assuming it's 1 - accuracy)
        failure_rate = 1 - accuracy
        
        # Calculate earliness (using the confidence threshold as a proxy)
        earliness = 5 * (1 - conf)  # Simplified calculation, adjust if needed
        
        # Add results
        metrics = [
            {'confidence': conf, 'value': tn, 'metric': 'tn'},
            {'confidence': conf, 'value': tp, 'metric': 'tp'},
            {'confidence': conf, 'value': fn, 'metric': 'fn'},
            {'confidence': conf, 'value': failure_rate, 'metric': 'failure_rate'},
            {'confidence': conf, 'value': fscore, 'metric': 'fscore'},
            {'confidence': conf, 'value': precision, 'metric': 'precision'},
            {'confidence': conf, 'value': fp, 'metric': 'fp'},
            {'confidence': conf, 'value': accuracy, 'metric': 'accuracy'},
            {'confidence': conf, 'value': earliness, 'metric': 'earliness'},
            {'confidence': conf, 'value': specificity, 'metric': 'specificity'},
            {'confidence': conf, 'value': recall, 'metric': 'recall'}
        ]
        detailed_results.extend(metrics)

# Convert to DataFrame and save
results_df = pd.DataFrame(detailed_results)
results_filepath = os.path.join(final_results_dir, "ohe_rf")
results_df.to_csv(results_filepath, sep=';', index=False)
print(f"Detailed results saved to: {results_filepath}")
