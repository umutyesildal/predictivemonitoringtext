import pandas as pd
from PredictiveModel import PredictiveModel
from sklearn.metrics import roc_auc_score, accuracy_score,classification_report

# 1) Read everything as string
df = pd.read_csv("data/BPI_Challenge_2017.csv", sep=";", low_memory=False, dtype=str)

# Print available columns
print("Available columns:", df.columns.tolist())

# 2) Convert times to numeric
df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
df["completeTime"] = pd.to_datetime(df["completeTime"], errors="coerce")
df["startTime"] = df["startTime"].astype("int64") // 10**9
df["completeTime"] = df["completeTime"].astype("int64") // 10**9

# 3) Convert label from e.g. "true"/"false" to 1/0
df["Accepted"] = df["Accepted"].str.lower()
df["Accepted"] = df["Accepted"].map({"true": 1, "false": 0})
df["Accepted"] = df["Accepted"].fillna(0).astype(int)

# 4) Convert other boolean columns, e.g., "Selected"
df["Selected"] = df["Selected"].str.lower()
df["Selected"] = df["Selected"].map({"true": 1, "false": 0})
df["Selected"] = df["Selected"].fillna(0).astype(int)

# 5) Convert numeric columns - Extended version
df["RequestedAmount"] = pd.to_numeric(df["RequestedAmount"], errors="coerce").fillna(0)
df["MonthlyCost"] = pd.to_numeric(df["MonthlyCost"], errors="coerce").fillna(0)
df["FirstWithdrawalAmount"] = pd.to_numeric(df["FirstWithdrawalAmount"], errors="coerce").fillna(0)
df["CreditScore"] = pd.to_numeric(df["CreditScore"], errors="coerce").fillna(0)
df["NumberOfTerms"] = pd.to_numeric(df["NumberOfTerms"], errors="coerce").fillna(0)
df["OfferedAmount"] = pd.to_numeric(df["OfferedAmount"], errors="coerce").fillna(0)

# Check data types after conversion
print("\nData types after conversion:")
numeric_cols = ["RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", "CreditScore", 
               "NumberOfTerms", "OfferedAmount"]
print(df[numeric_cols].dtypes)

# 6) Sort & reindex events
df.sort_values(by=["case", "startTime"], inplace=True)
df["event"] = df.groupby("case").cumcount() + 1

# 7) Train/test split
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# 8) Define your encoder_kwargs with new encoding method
encoder_kwargs = {
    "event_nr_col": "event",
    "static_cols": ["RequestedAmount", "MonthlyCost", "FirstWithdrawalAmount", "CreditScore"],
    "dynamic_cols": ["startTime", "completeTime"],
    "cat_cols": ["ApplicationType", "LoanGoal", "EventOrigin", "Action"],
    "encoding_method": "onehot",
    "oversample_fit": False,
    "minority_label": "0",
    "fillna": True,
    "random_state": 22
}

# Check data types and missing values
print("\nData types:")
print(train_df[encoder_kwargs["static_cols"] + encoder_kwargs["dynamic_cols"] + encoder_kwargs["cat_cols"]].dtypes)

print("\nMissing values:")
print(train_df[encoder_kwargs["static_cols"] + encoder_kwargs["dynamic_cols"] + encoder_kwargs["cat_cols"]].isnull().sum())

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
