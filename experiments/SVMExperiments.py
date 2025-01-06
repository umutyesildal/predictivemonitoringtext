import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from PredictiveMonitor import PredictiveMonitor

# Data loading and preprocessing
data_filepath = os.path.join(parent_dir, "data", "BPI_Challenge_2017_unstructured.csv")
data = pd.read_csv(data_filepath, sep=";", encoding="utf-8", low_memory=False)

# Take a sample of the data for initial testing
unique_cases = data['case'].unique()
sample_size = 500  # Increased from 100 to 500 for better representation
sampled_cases = pd.Series(unique_cases).sample(n=min(sample_size, len(unique_cases)), random_state=22)
data = data[data['case'].isin(sampled_cases)]

print(f"\nUsing {len(sampled_cases)} cases from the dataset")

# Print column names to see what we have
print("Available columns:", data.columns.tolist())

# Basic preprocessing
print("\nStarting preprocessing...")

# Convert EventDescription to lowercase for text processing
data['event_description_lemmas'] = data['EventDescription'].str.lower()

# Add necessary columns if they don't exist
if 'case_name' not in data.columns:
    data['case_name'] = data['case'].astype(str)
    
if 'label' not in data.columns:
    # Print unique values in Action and Accepted columns to understand the data
    print("\nUnique values in Action column:", data['Action'].unique())
    print("Action value counts:", data['Action'].value_counts())
    print("\nUnique values in Accepted column:", data['Accepted'].unique())
    print("Accepted value counts:", data['Accepted'].value_counts())
    
    # Create label based on 'Accepted' column and handle any missing values
    data['label'] = data.groupby('case')['Accepted'].transform(lambda x: 'successful' if any(x == True) else 'unsuccessful')
    
    # Print label distribution for debugging
    print("\nLabel distribution:", data['label'].value_counts())
    print("Number of unique cases:", data['case'].nunique())
    print("Number of total rows:", len(data))

# Add event_nr column if it doesn't exist
if 'event_nr' not in data.columns:
    # Group by case and add event number
    data['event_nr'] = data.groupby('case_name').cumcount() + 1
    print("\nEvent number distribution:", data['event_nr'].value_counts().sort_index().head())

# Add case_length column
data['case_length'] = data.groupby('case_name')['event_nr'].transform('max')
print("\nCase length statistics:")
print(data.groupby('case_name')['case_length'].first().describe())

# Print some example cases for verification
print("\nExample cases:")
example_case = data['case_name'].iloc[0]
print(f"\nDetails for case {example_case}:")
print(data[data['case_name'] == example_case][['event_nr', 'case_length', 'label', 'Accepted', 'Action']].head())

# Create cv_results directory if it doesn't exist
cv_results_dir = os.path.join(current_dir, "cv_results")
if not os.path.exists(cv_results_dir):
    os.makedirs(cv_results_dir)

# Make sure output files have .csv extension
def get_output_filename(base_name):
    return os.path.join(cv_results_dir, f"{base_name}.csv")

# Save preprocessed data
preprocessed_filepath = os.path.join(parent_dir, "data", "data_preprocessed.csv")
data.to_csv(preprocessed_filepath, sep=";", encoding="utf-8", index=False)

# Define columns for the model
dynamic_cols = ["RequestedAmount", "MonthlyCost", "NumberOfTerms", "CreditScore", "OfferedAmount", "event_description_lemmas"]
static_cols = ["case_name", "label", "ApplicationType", "LoanGoal", "case_length"]
cat_cols = ["ApplicationType", "LoanGoal"]

case_id_col = "case_name"
label_col = "label"
event_nr_col = "event_nr"
text_col = "event_description_lemmas"
pos_label = "unsuccessful"

# divide into train and test data
train_names, test_names = train_test_split(data[case_id_col].unique(), train_size=4.0/5, random_state=22)
train = data[data[case_id_col].isin(train_names)]
test = data[data[case_id_col].isin(test_names)]

# Print training data label distribution
print("\nTraining data label distribution:", train[train[event_nr_col]==1][label_col].value_counts())

# create three folds for cross-validation out of training data (reduced from 5 to 3)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

# Adjust parameters for better results
confidences = [0.2, 0.25, 0.3]  # Keep existing confidence levels
text_transformer_type = "BoNGTransformer"
cls_method = "svm"

encoder_kwargs = {
    "event_nr_col": event_nr_col,
    "static_cols": static_cols,
    "dynamic_cols": dynamic_cols,
    "cat_cols": cat_cols,
    "oversample_fit": True,
    "minority_label": pos_label,
    "fillna": True,
    "random_state": 22
}

part = 1
# Get the data for first events of each case for stratification
first_events = train[train[event_nr_col]==1]
print(f"\nFirst events shape: {first_events.shape}")
print("First events label distribution:", first_events[label_col].value_counts())

for train_index, test_index in kf.split(first_events, first_events[label_col]):
    print(f"\nProcessing fold {part}/5")
    
    # create train and validation data according to current fold
    current_train_names = first_events.iloc[train_index][case_id_col].values
    train_chunk = train[train[case_id_col].isin(current_train_names)]
    current_test_names = first_events.iloc[test_index][case_id_col].values
    test_chunk = train[train[case_id_col].isin(current_test_names)]
    
    print(f"Train chunk shape: {train_chunk.shape}")
    print(f"Test chunk shape: {test_chunk.shape}")
    print("Train chunk label distribution:", train_chunk[train_chunk[event_nr_col]==1][label_col].value_counts())
    print("Test chunk label distribution:", test_chunk[test_chunk[event_nr_col]==1][label_col].value_counts())
    
    # Focused parameter combinations based on previous results
    for kernel in ['rbf']:  # Keep RBF as it performed best
        for C in [0.5, 1.0, 2.0]:  # More granular C values around 1.0
            cls_kwargs = {
                "kernel": kernel,
                "C": C,
                "random_state": 22,
                "probability": True,
                "class_weight": {pos_label: 2}  # Explicit class weight for minority class
            }
            
            for nr_selected in [400, 500, 600]:  # Increased feature counts
                for tfidf in [True]:  # Keep TFIDF as it worked well
                    for ngram_max in [1, 2]:  # Keep both unigrams and bigrams
                        print(f"\nTraining model with: kernel={kernel}, C={C}, features={nr_selected}, ngram={ngram_max}")
                        
                        transformer_kwargs = {
                            "ngram_max": ngram_max,
                            "tfidf": tfidf,
                            "nr_selected": nr_selected
                        }
                        
                        # train
                        predictive_monitor = PredictiveMonitor(
                            event_nr_col=event_nr_col,
                            case_id_col=case_id_col,
                            label_col=label_col,
                            pos_label=pos_label,
                            encoder_kwargs=encoder_kwargs,
                            cls_kwargs=cls_kwargs,
                            transformer_kwargs=transformer_kwargs,
                            text_col=text_col,
                            text_transformer_type=text_transformer_type,
                            cls_method=cls_method
                        )
                        
                        try:
                            predictive_monitor.train(train_chunk)
                            
                            # test
                            base_filename = f"svm_{kernel}_C{C}_selected{nr_selected}_{('tfidf_' if tfidf else '')}ngram{ngram_max}_part{part}"
                            output_filename = get_output_filename(base_filename)
                            predictive_monitor.test(test_chunk, confidences, evaluate=True, output_filename=output_filename)
                            
                        except Exception as e:
                            print(f"Error during training/testing: {str(e)}")
                            continue
    
    part += 1

# Print summary of results
print("\n=== RESULTS SUMMARY ===")
all_results = {}
for filename in os.listdir(cv_results_dir):
    if filename.startswith('svm_') and filename.endswith('.csv'):
        # Extract configuration from filename
        config = filename.replace('.csv', '')
        results_df = pd.read_csv(os.path.join(cv_results_dir, filename), sep=';')
        
        # Group results by confidence level
        for confidence in confidences:
            conf_results = results_df[results_df['confidence'] == confidence]
            
            # Skip if no results for this confidence level
            if len(conf_results) == 0:
                print(f"No results found for {config} at confidence {confidence}")
                continue
                
            try:
                # Store results in dictionary
                key = f"{config}_conf{confidence}"
                all_results[key] = {
                    'accuracy': conf_results[conf_results['metric'] == 'accuracy']['value'].iloc[0] if not conf_results[conf_results['metric'] == 'accuracy'].empty else 0.0,
                    'precision': conf_results[conf_results['metric'] == 'precision']['value'].iloc[0] if not conf_results[conf_results['metric'] == 'precision'].empty else 0.0,
                    'recall': conf_results[conf_results['metric'] == 'recall']['value'].iloc[0] if not conf_results[conf_results['metric'] == 'recall'].empty else 0.0,
                    'f1': conf_results[conf_results['metric'] == 'f1']['value'].iloc[0] if not conf_results[conf_results['metric'] == 'f1'].empty else 0.0,
                    'earliness': conf_results[conf_results['metric'] == 'earliness']['value'].iloc[0] if not conf_results[conf_results['metric'] == 'earliness'].empty else 0.0
                }
            except Exception as e:
                print(f"Error processing results for {config} at confidence {confidence}: {str(e)}")
                continue

# Print results in a formatted way
if len(all_results) == 0:
    print("\nNo results found. Check if the models were trained and tested successfully.")
else:
    for confidence in confidences:
        print(f"\nResults for confidence level {confidence}:")
        print("Configuration | Accuracy | Precision | Recall | F1 | Earliness")
        print("-" * 80)
        
        # Filter results for this confidence level
        conf_results = {k: v for k, v in all_results.items() if f"_conf{confidence}" in k}
        
        if len(conf_results) == 0:
            print(f"No results found for confidence level {confidence}")
            continue
            
        for config, metrics in conf_results.items():
            # Clean up config name for display
            clean_config = config.replace(f"_conf{confidence}", "")
            print(f"{clean_config} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['earliness']:.3f}")

# Print overall statistics
print("\n=== OVERALL STATISTICS ===")
print(f"Total configurations tested: {len(all_results) // len(confidences)}")
print(f"Total result files: {len([f for f in os.listdir(cv_results_dir) if f.startswith('svm_') and f.endswith('.csv')])}")

# Add final evaluation on test set
print("\nFinal Evaluation on Test Set:")
print("Encoding training data...")

# Train final model with best configuration
best_cls_kwargs = {
    "kernel": "rbf",
    "C": 1.0,
    "random_state": 22,
    "probability": True,
    "class_weight": {pos_label: 2}
}

best_transformer_kwargs = {
    "ngram_max": 1,
    "tfidf": True,
    "nr_selected": 500
}

final_model = PredictiveMonitor(
    event_nr_col=event_nr_col,
    case_id_col=case_id_col,
    label_col=label_col,
    pos_label=pos_label,
    encoder_kwargs=encoder_kwargs,
    cls_kwargs=best_cls_kwargs,
    transformer_kwargs=best_transformer_kwargs,
    text_col=text_col,
    text_transformer_type=text_transformer_type,
    cls_method=cls_method
)

try:
    final_model.train(train)
    final_preds = final_model.predict(test)
    final_probs = final_model.predict_proba(test)
    
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    
    # Calculate and print metrics
    accuracy = accuracy_score(final_model.test_y, final_preds)
    auc = roc_auc_score(final_model.test_y, final_probs[:, 1])
    
    print(f"\nTest Set Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC: {auc:.3f}")
    
    # Generate detailed classification report
    report = classification_report(final_model.test_y, final_preds, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Format the dataframe for better display
    df_report = df_report.round(3)
    df_report['support'] = df_report['support'].astype(int)
    
    # Print formatted report
    print("\nClassification Report:")
    print("               precision    recall  f1-score  support")
    print("-" * 55)
    for idx in df_report.index:
        if idx in ['successful', 'unsuccessful']:  # Class specific metrics
            print(f"{idx:>16} {df_report.loc[idx, 'precision']:>9.3f} {df_report.loc[idx, 'recall']:>8.3f} {df_report.loc[idx, 'f1-score']:>9.3f} {df_report.loc[idx, 'support']:>8}")
        elif idx == 'accuracy':  # Overall accuracy
            print("-" * 55)
            print(f"{'accuracy':>16} {df_report.loc[idx, 'precision']:>9.3f} {' ':>8} {' ':>9} {df_report.loc[idx, 'support']:>8}")
        elif idx in ['macro avg', 'weighted avg']:  # Average metrics
            print(f"{idx:>16} {df_report.loc[idx, 'precision']:>9.3f} {df_report.loc[idx, 'recall']:>8.3f} {df_report.loc[idx, 'f1-score']:>9.3f} {df_report.loc[idx, 'support']:>8}")
    
    # Save the final report
    report_filepath = os.path.join(cv_results_dir, 'final_svm_classification_report.csv')
    df_report.to_csv(report_filepath)
    print(f"\nDetailed report saved to: {report_filepath}")
    
except Exception as e:
    print(f"Error during final evaluation: {str(e)}")
