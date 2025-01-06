import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
import sys
sys.path.append('..')
from PredictiveMonitor import PredictiveMonitor

# Add parent directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Create results directory if it doesn't exist
results_dir = os.path.join(current_dir, "loo_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

data_filepath = os.path.join(parent_dir, "data", "BPI_Challenge_2017_unstructured.csv")
data = pd.read_csv(data_filepath, sep=";", encoding="utf-8", low_memory=False)

static_cols = [
    "RequestedAmount", "MonthlyCost", "CreditScore", "NumberOfTerms", "Accepted"
]
dynamic_cols = ["startTime", "completeTime"]
cat_cols = ["ApplicationType", "LoanGoal"]  # Using most important categorical columns
case_id_col = "case"
label_col = "Accepted"
event_nr_col = "event"
last_state_cols = ["EventDescription"]
text_col = "EventDescription"
pos_label = "unsuccessful"

# divide into train and test data
train_names, test_names = train_test_split(data[case_id_col].unique(), train_size=0.25, random_state=22)
train = data[data[case_id_col].isin(train_names)]
test = data[data[case_id_col].isin(test_names)]

# create LeaveOneOut cross-validation
loo = LeaveOneOut()

# Convert event_nr_col to integers for max_events calculation
train[event_nr_col] = train[event_nr_col].astype('category').cat.codes + 1

confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
text_transformer_type = "BoNGTransformer"
cls_methods = ["rf", "logit"]

encoder_kwargs = {"event_nr_col":event_nr_col, "static_cols":static_cols, "dynamic_cols":dynamic_cols,
                  "last_state_cols":last_state_cols, "cat_cols":cat_cols, "oversample_fit":True, 
                  "minority_label":pos_label, "fillna":True, "random_state":22}

part = 1
filtered_train = train[train[event_nr_col] == 1]
if not filtered_train.empty:
    filtered_train_indices = filtered_train.index
    for train_index, test_index in loo.split(filtered_train[label_col]):
        
        # create train and validation data according to current fold
        current_train_names = filtered_train_indices[train_index]
        train_chunk = train.loc[current_train_names]
        current_test_names = filtered_train_indices[test_index]
        test_chunk = train.loc[current_test_names]
        
        for cls_method in cls_methods:
            if cls_method == "rf":
                cls_kwargs = {"n_estimators":500, "random_state":22}
            else:
                cls_kwargs = {"random_state":22}
                
            for nr_selected in [100, 250, 500, 750, 1000, 2000, 5000]:
                for tfidf in [True, False]:
                    for ngram_max in [1,2,3]:
                        
                        transformer_kwargs = {"ngram_max":ngram_max, "tfidf":tfidf, "nr_selected":nr_selected}
                        
                        # train
                        predictive_monitor = PredictiveMonitor(event_nr_col=event_nr_col, case_id_col=case_id_col, label_col=label_col, pos_label=pos_label, encoder_kwargs=encoder_kwargs, cls_kwargs=cls_kwargs, transformer_kwargs=transformer_kwargs, text_col=text_col, text_transformer_type=text_transformer_type, cls_method=cls_method)
                        predictive_monitor.train(train_chunk)
                        
                        # test
                        predictive_monitor.test(test_chunk, confidences, evaluate=True, output_filename=os.path.join(results_dir, "bong_selected%s_%sngram%s_%s_part%s.csv" % (nr_selected, ("tfidf_" if tfidf else ""), ngram_max, cls_method, part)))
                        
        part += 1
else:
    print("No samples found for the specified event number.")