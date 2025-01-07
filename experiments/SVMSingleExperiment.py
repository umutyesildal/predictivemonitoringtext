import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

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
sample_size = 5000  # Using more data for better learning
sampled_cases = pd.Series(unique_cases).sample(n=min(sample_size, len(unique_cases)), random_state=22)
data = data[data['case'].isin(sampled_cases)]

print(f"\nUsing {len(sampled_cases)} cases from the dataset")

# Basic preprocessing
print("\nStarting preprocessing...")

# Convert EventDescription to lowercase for text processing
data['event_description_lemmas'] = data['EventDescription'].str.lower()

# Add necessary columns
data['case_name'] = data['case'].astype(str)
data['label'] = data.groupby('case')['Accepted'].transform(lambda x: 'successful' if any(x == True) else 'unsuccessful')
data['event_nr'] = data.groupby('case_name').cumcount() + 1
data['case_length'] = data.groupby('case_name')['event_nr'].transform('max')

# Print label distribution
print("\nLabel distribution:", data['label'].value_counts())
print("Number of unique cases:", data['case'].nunique())
print("Number of total rows:", len(data))

# Define columns for the model
dynamic_cols = ["RequestedAmount", "MonthlyCost", "NumberOfTerms", "CreditScore", "OfferedAmount", "event_description_lemmas"]
static_cols = ["case_name", "label", "ApplicationType", "LoanGoal", "case_length"]
cat_cols = ["ApplicationType", "LoanGoal"]

case_id_col = "case_name"
label_col = "label"
event_nr_col = "event_nr"
text_col = "event_description_lemmas"
pos_label = "unsuccessful"

# Split into train and test
train_names, test_names = train_test_split(data[case_id_col].unique(), train_size=4.0/5, random_state=22)
train = data[data[case_id_col].isin(train_names)]
test = data[data[case_id_col].isin(test_names)]

print("\nTrain set shape:", train.shape)
print("Test set shape:", test.shape)

# Best configuration parameters
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

cls_kwargs = {
    "kernel": "rbf",
    "C": 1.0,
    "random_state": 22,
    "probability": True,
    "class_weight": {pos_label: 2}
}

transformer_kwargs = {
    "ngram_max": 1,
    "tfidf": True,
    "nr_selected": 500
}

# Initialize and train model
print("\nTraining model with best configuration...")
model = PredictiveMonitor(
    event_nr_col=event_nr_col,
    case_id_col=case_id_col,
    label_col=label_col,
    pos_label=pos_label,
    encoder_kwargs=encoder_kwargs,
    cls_kwargs=cls_kwargs,
    transformer_kwargs=transformer_kwargs,
    text_col=text_col,
    text_transformer_type="BoNGTransformer",
    cls_method="svm"
)

# Train and evaluate
print("Fitting model...")
model.train(train)
print("Model fitting completed.")

print("\nMaking predictions...")
confidences = [0.2, 0.25, 0.3, 0.35]
results = model.test(test, confidences=confidences, evaluate=True)

# Print results for each confidence level
for conf, metrics in results.items():
    print(f"\nResults for confidence level {conf}:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"Earliness: {metrics['earliness']:.3f}")

# Create results directory if it doesn't exist
results_dir = os.path.join(current_dir, "single_test_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save detailed results
output_filename = os.path.join(results_dir, 'svm_single_test_results.csv')
with open(output_filename, 'w') as fout:
    fout.write("confidence;value;metric\n")
    for conf, metrics in results.items():
        for metric, value in metrics.items():
            fout.write(f"{conf};{value};{metric}\n")

print(f"\nDetailed results saved to: {output_filename}")
