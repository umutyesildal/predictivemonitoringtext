import pandas as pd
from PredictiveModel import PredictiveModel
from sklearn.metrics import roc_auc_score, accuracy_score,classification_report

# 1) Read everything as string
df = pd.read_csv("data/BPI_Challenge_2017.csv", sep=";", low_memory=False, dtype=str)

# 2) Convert times to numeric
df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
df["completeTime"] = pd.to_datetime(df["completeTime"], errors="coerce")
df["startTime"] = df["startTime"].view("int64") // 10**9  # or .astype("int64") / 1e9
df["completeTime"] = df["completeTime"].view("int64") // 10**9

# 3) Convert label from e.g. "true"/"false" to 1/0
df["Accepted"] = df["Accepted"].str.lower()
df["Accepted"] = df["Accepted"].map({"true": 1, "false": 0})
df["Accepted"] = df["Accepted"].fillna(0).astype(int)

# 4) Convert other boolean columns, e.g., "Selected"
df["Selected"] = df["Selected"].str.lower()
df["Selected"] = df["Selected"].map({"true": 1, "false": 0})
df["Selected"] = df["Selected"].fillna(0).astype(int)

# 5) Convert numeric columns
df["RequestedAmount"] = pd.to_numeric(df["RequestedAmount"], errors="coerce").fillna(0)
df["MonthlyCost"] = pd.to_numeric(df["MonthlyCost"], errors="coerce").fillna(0)
# etc. for other numeric columns: CreditScore, NumberOfTerms, OfferedAmount, etc.

# 6) Sort & reindex events
df.sort_values(by=["case", "startTime"], inplace=True)
df["event"] = df.groupby("case").cumcount() + 1

# 7) Train/test split
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# 8) Define your encoder_kwargs, transformer_kwargs, cls_kwargs as before
encoder_kwargs = {
    "event_nr_col": "event",
    "static_cols": ["ApplicationType", "LoanGoal", "RequestedAmount"],
    "dynamic_cols": ["startTime", "completeTime", "MonthlyCost"],
    "cat_cols": ["ApplicationType", "LoanGoal"],
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

cls_kwargs = {
    "n_estimators": 100,
    "random_state": 22
}

model = PredictiveModel(
    nr_events=3,
    case_id_col="case",
    label_col="Accepted",
    text_col="text",                
    text_transformer_type=None,
    cls_method="rf",
    encoder_kwargs=encoder_kwargs,
    transformer_kwargs=transformer_kwargs,
    cls_kwargs=cls_kwargs
)

model.fit(train_df)
preds_proba = model.predict_proba(test_df)
print("Predicted probability shape:", preds_proba.shape)



# Suppose your true labels are in model.test_y
y_true = model.test_y  # or however your pipeline stores it

# If you want a hard label:
preds_label = preds_proba.argmax(axis=1)  # picks the class with highest probability

print("Accuracy:", accuracy_score(y_true, preds_label))
print("AUC:", roc_auc_score(y_true, preds_proba[:,1]))  # probability for class 1
print(classification_report(y_true, preds_label))
