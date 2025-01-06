from SequenceEncoder import SequenceEncoder
from TextTransformers import LDATransformer, PVTransformer, BoNGTransformer, NBLogCountRatioTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import numpy as np

class PredictiveModel():

    def __init__(
        self, nr_events, case_id_col, label_col, text_col=None, text_transformer_type=None,
        cls_method="rf", encoder_kwargs=None, transformer_kwargs=None, cls_kwargs=None
    ):
        """
        PredictiveModel for a SINGLE prefix length (nr_events).
        
        Args:
            nr_events (int): fixed prefix length (1, 2, 3, ...).
            case_id_col (str): name of the column with case (process-instance) IDs.
            label_col (str): name of the outcome/label column.
            encoder_kwargs (dict): passed to SequenceEncoder constructor.
            transformer_kwargs (dict): passed to the selected TextTransformer.
            cls_kwargs (dict): passed to the classifier constructor.
            text_col (str): name of the text column(s) prefix (if any).
            text_transformer_type (str): one of the text transformers {None, "LDATransformer", ...}.
            cls_method (str): "rf" for RandomForestClassifier, "logit" for LogisticRegression.
        """
        self.nr_events = nr_events
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.text_col = text_col
        self.text_transformer_type = text_transformer_type
        self.cls_method = cls_method
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.transformer_kwargs = transformer_kwargs if transformer_kwargs is not None else {}
        self.cls_kwargs = cls_kwargs if cls_kwargs is not None else {}
        
        # Store static_cols from encoder_kwargs
        self.static_cols = self.encoder_kwargs.get('static_cols', [])
        
        # Initialize classifier
        if self.cls_method == "rf":
            self.cls = RandomForestClassifier(**self.cls_kwargs)
        elif self.cls_method == "logit":
            self.cls = LogisticRegression(**self.cls_kwargs)
            self.scaler = StandardScaler()
        elif self.cls_method == "svm":
            self.cls = SVC(**self.cls_kwargs)
            self.scaler = StandardScaler()
        
        # Initialize encoder
        self.encoder = SequenceEncoder(
            nr_events=self.nr_events,
            case_id_col=self.case_id_col,
            label_col=self.label_col,
            **self.encoder_kwargs
        )
        
        # Initialize text transformer if specified
        if self.text_transformer_type == "bong":
            self.transformer = BoNGTransformer(**self.transformer_kwargs)
        else:
            self.transformer = None
        
        self.hardcoded_prediction = None
        self.test_encode_time = None
        self.test_preproc_time = None
        self.test_time = None
        self.nr_test_cases = None

    def fit(self, dt_train):
        print("Encoding training data...")
        train_encoded = self.encoder.fit_transform(dt_train)
        
        print("Preparing features and target...")
        self.train_X = train_encoded.drop([self.case_id_col, self.label_col], axis=1)
        self.train_y = train_encoded[self.label_col]
        
        # Calculate class weights
        if hasattr(self.cls, 'class_weight'):
            n_samples = len(self.train_y)
            n_classes = len(np.unique(self.train_y))
            class_weights = dict(enumerate(n_samples / (n_classes * np.bincount(self.train_y))))
            self.cls.class_weight = class_weights
            print("\nClass weights:", class_weights)
        
        print("\nFeature Value Distribution:")
        feature_non_zero = {}
        for col in self.train_X.columns:
            non_zero = (self.train_X[col] != 0).sum()
            feature_non_zero[col] = non_zero/len(self.train_X)
            print(f"{col}: {non_zero} non-zero values ({feature_non_zero[col]*100:.2f}%)")
        
        # Remove features with very low non-zero values
        low_value_features = [col for col, ratio in feature_non_zero.items() 
                            if ratio < 0.01 and col not in self.static_cols]
        if low_value_features:
            print(f"\nRemoving {len(low_value_features)} features with < 1% non-zero values:")
            print(low_value_features)
            self.train_X = self.train_X.drop(columns=low_value_features)
        
        # Text transformation should be done before scaling
        if self.transformer is not None:
            print("\nText Transformation Details:")
            print(f"Transformer type: {type(self.transformer).__name__}")
            text_cols = [col for col in self.train_X.columns if col.startswith(self.text_col)]
            print(f"Found {len(text_cols)} text columns")
            
            if text_cols:
                # Create a DataFrame with just the text columns
                text_data = self.train_X[text_cols].copy()
                print(f"Text data shape before transformation: {text_data.shape}")
                
                # Convert to string and fill NaN values
                for col in text_cols:
                    text_data[col] = text_data[col].fillna('').astype(str)
                    print(f"Sample values from {col}:")
                    print(text_data[col].head())
                
                # Transform text data
                train_text = self.transformer.fit_transform(text_data, self.train_y)
                print(f"Text data shape after transformation: {train_text.shape}")
                
                # Combine with non-text features
                self.train_X = pd.concat([self.train_X.drop(text_cols, axis=1), train_text], axis=1)
                print(f"Final feature matrix shape: {self.train_X.shape}")
        
        # Convert all remaining object/string columns to numeric
        for col in self.train_X.select_dtypes(include=['object']).columns:
            self.train_X[col] = pd.Categorical(self.train_X[col]).codes
        
        # Feature scaling for SVM
        if hasattr(self, 'scaler'):
            print("Scaling features...")
            self.train_X = pd.DataFrame(
                self.scaler.fit_transform(self.train_X),
                columns=self.train_X.columns,
                index=self.train_X.index
            )
        
        print("Fitting classifier...")
        self.cls.fit(self.train_X, self.train_y)
        
        if hasattr(self.cls, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.train_X.columns,
                'importance': self.cls.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 20 Most Important Features:")
            print(importances.head(20))
        
        print("Model fitting completed.")
        
        return self

    def predict_proba(self, dt_test):
        """
        Predict class probabilities for the test dataset.
        1) Sequence-encode the test data
        2) Transform text columns
        3) Scale features
        4) Return predicted probabilities
        """
        encode_start_time = time.time()
        test_encoded = self.encoder.transform(dt_test)
        encode_end_time = time.time()
        self.test_encode_time = encode_end_time - encode_start_time
        
        # Check if encoding returned an empty DataFrame
        if test_encoded.empty:
            print("Warning: after sequence encoding, test data is empty. Returning empty predictions.")
            return np.array([])  # No data to predict on
        
        test_preproc_start_time = time.time()
        
        # Prepare features (X)
        test_X = test_encoded.drop([self.case_id_col, self.label_col], axis=1)
        test_y = test_encoded[self.label_col]  # We might need this for some transformers
        
        # Text transformation
        if self.transformer is not None:
            text_cols = [col for col in test_X.columns if col.startswith(self.text_col)]
            if text_cols:
                # Create a DataFrame with just the text columns
                text_data = test_X[text_cols].copy()
                # Convert to string and fill NaN values
                for col in text_cols:
                    text_data[col] = text_data[col].fillna('').astype(str)
                # Transform text data
                test_text = self.transformer.transform(text_data)
                # Combine with non-text features
                test_X = pd.concat([test_X.drop(text_cols, axis=1), test_text], axis=1)
        
        # Convert all remaining object/string columns to numeric
        for col in test_X.select_dtypes(include=['object']).columns:
            test_X[col] = pd.Categorical(test_X[col]).codes
        
        # Ensure test features match training features
        missing_cols = set(self.train_X.columns) - set(test_X.columns)
        extra_cols = set(test_X.columns) - set(self.train_X.columns)
        
        if missing_cols or extra_cols:
            print("\nAdjusting feature columns to match training data:")
            if missing_cols:
                print(f"Adding {len(missing_cols)} missing columns with zeros")
                for col in missing_cols:
                    test_X[col] = 0
            if extra_cols:
                print(f"Removing {len(extra_cols)} extra columns")
                test_X = test_X.drop(columns=extra_cols)
        
        # Reorder columns to match training data
        test_X = test_X.reindex(columns=self.train_X.columns, fill_value=0)
        
        # Feature scaling for SVM
        if hasattr(self, 'scaler'):
            test_X = pd.DataFrame(
                self.scaler.transform(test_X),
                columns=test_X.columns,
                index=test_X.index
            )
        
        self.test_case_names = test_encoded[self.case_id_col]
        self.test_X = test_X
        self.test_y = test_encoded[self.label_col]
        
        test_preproc_end_time = time.time()
        self.test_preproc_time = test_preproc_end_time - test_preproc_start_time
        
        test_start_time = time.time()
        predictions_proba = self.cls.predict_proba(test_X)
        test_end_time = time.time()
        self.test_time = test_end_time - test_start_time
        self.nr_test_cases = len(predictions_proba)
        
        return predictions_proba
