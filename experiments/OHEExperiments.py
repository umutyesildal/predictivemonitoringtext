import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys
import os
import numpy as np
from sklearn.utils import resample

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
    print("\nPreprocessing Data:")
    print(f"Initial shape: {df.shape}")
    
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
    
    # Print some event statistics
    print("\nEvent Statistics:")
    event_counts = df.groupby("event").size()
    print(event_counts.head(10))
    
    print("\nSample of first few events for a case:")
    sample_case = df["case"].iloc[0]
    print(df[df["case"] == sample_case][["case", "event", "Action", "startTime"]].head())
    
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

# Print some statistics about the data
print("\nData Statistics:")
print(f"Total number of events: {len(data):,}")
print(f"Number of unique cases: {data[case_id_col].nunique():,}")
print(f"Average events per case: {len(data) / data[case_id_col].nunique():.2f}")
print(f"Label distribution:\n{data.groupby(case_id_col)[label_col].first().value_counts(normalize=True)}")

# Print unique values in categorical columns
print("\nUnique values in categorical columns:")
for col in cat_cols + [event_col]:
    print(f"\n{col} unique values:")
    print(data[col].value_counts().head())

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

# Print split statistics
print("\nSplit Statistics:")
print(f"Training set cases: {len(train_names)}")
print(f"Test set cases: {len(test_names)}")
print(f"Training set events: {len(train)}")
print(f"Test set events: {len(test)}")

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
    
    # Create train and validation data for current fold
    current_train_names = train_names_array[train_index]
    train_chunk = train[train[case_id_col].isin(current_train_names)]
    current_test_names = train_names_array[test_index]
    test_chunk = train[train[case_id_col].isin(current_test_names)]
    
    for cls_method in cls_methods:
        # Define classifier parameters
        if cls_method == "rf":
            cls_kwargs = {"n_estimators": 100, "random_state": 22}
        else:
            cls_kwargs = {"random_state": 22}
        
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
        
        # Print debug information before training
        print("\nModel Configuration:")
        print(f"Number of events: 5")
        print(f"Text transformer: BoNGTransformer")
        print(f"Text transformer parameters:")
        print(f"  - ngram_min: {transformer_kwargs['ngram_min']}")
        print(f"  - ngram_max: {transformer_kwargs['ngram_max']}")
        print(f"  - tfidf: {transformer_kwargs['tfidf']}")
        print(f"  - nr_selected: {transformer_kwargs['nr_selected']}")
        print(f"Classifier: {cls_method}")
        print(f"Static columns: {static_cols}")
        print(f"Dynamic columns: {dynamic_cols}")
        print(f"Categorical columns: {cat_cols}")
        print(f"Event column: {event_col}")

        # Train model
        model.fit(train_chunk)

        # Get training features and target
        train_encoded = model.encoder.transform(train_chunk)
        train_X = train_encoded.drop([case_id_col, label_col], axis=1)

        # Print feature importance if using RandomForest
        if cls_method == "rf":
            try:
                # Get feature names and importance scores
                feature_names = model.train_X.columns.tolist()
                importance_scores = model.cls.feature_importances_
                
                if len(feature_names) == len(importance_scores):
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance_scores
                    }).sort_values('importance', ascending=False)
                    
                    print("\nTop 10 Most Important Features:")
                    print(feature_importance.head(10))
                    
                    # Save feature importance to CSV
                    importance_file = os.path.join(results_dir, f"feature_importance_fold{part}.csv")
                    feature_importance.to_csv(importance_file)
                    print(f"\nSaved feature importance to: {importance_file}")
                else:
                    print("\nWarning: Feature names and importance scores have different lengths")
                    print(f"Number of features: {len(feature_names)}")
                    print(f"Number of importance scores: {len(importance_scores)}")
            except Exception as e:
                print(f"\nError calculating feature importance: {str(e)}")

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
                
                # Add safety check for AUC calculation
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
    else:
        print("AUC: Not calculable (only one class present)")
except IndexError:
    print("AUC: Not calculable (prediction error)")
print("\nClassification Report:")
print(classification_report(final_model.test_y, final_preds))
