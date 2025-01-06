from SequenceEncoder import SequenceEncoder
from TextTransformers import LDATransformer, PVTransformer, BoNGTransformer, NBLogCountRatioTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

import pandas as pd
import time
import numpy as np

class PredictiveModel():

    def __init__(
        self, nr_events, case_id_col, label_col, 
        encoder_kwargs, transformer_kwargs, cls_kwargs, 
        text_col=None, text_transformer_type=None, cls_method="rf"
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
        self.text_col = text_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        
        # Sequence Encoder (for numeric event sequences)
        self.encoder = SequenceEncoder(
            nr_events=nr_events, 
            case_id_col=case_id_col, 
            label_col=label_col,
            **encoder_kwargs
        )
        
        # Text Transformer
        if text_transformer_type is None:
            self.transformer = None
        elif text_transformer_type == "LDATransformer":
            self.transformer = LDATransformer(**transformer_kwargs)
        elif text_transformer_type == "BoNGTransformer":
            self.transformer = BoNGTransformer(**transformer_kwargs)
        elif text_transformer_type == "NBLogCountRatioTransformer":
            self.transformer = NBLogCountRatioTransformer(**transformer_kwargs)
        elif text_transformer_type == "PVTransformer":
            self.transformer = PVTransformer(**transformer_kwargs)
        else:
            print("Transformer type not known")
            self.transformer = None

        # Classifier
        if cls_method == "logit":
            self.cls = LogisticRegression(**cls_kwargs)
        elif cls_method == "rf":
            self.cls = RandomForestClassifier(**cls_kwargs)
        elif cls_method == "svm":
            self.cls = SVC(probability=True, **cls_kwargs)
        else:
            print("Classifier method not known. Defaulting to RandomForest.")
            self.cls = RandomForestClassifier(**cls_kwargs)
        
        self.hardcoded_prediction = None
        self.test_encode_time = None
        self.test_preproc_time = None
        self.test_time = None
        self.nr_test_cases = None

    def fit(self, dt_train):
        print("Encoding training data...")
        train_encoded = self.encoder.fit_transform(dt_train)
        
        print("Preparing features and target...")
        train_X = train_encoded.drop([self.case_id_col, self.label_col], axis=1)
        train_y = train_encoded[self.label_col]
        
        print("Fitting classifier...")
        self.cls.fit(train_X, train_y)
        print("Model fitting completed.")
        
        return self

    def predict_proba(self, dt_test):
        """
        Predict class probabilities for the test dataset.
        1) Sequence-encode the test data
        2) Optionally transform text columns
        3) Return predicted probabilities
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
        
        # Text transformation
        if self.transformer is not None:
            text_cols = [col for col in test_X.columns if col.startswith(self.text_col)]
            for col in text_cols:
                test_X[col] = test_X[col].astype('str')
            test_text = self.transformer.transform(test_X[text_cols])
            test_X = pd.concat([test_X.drop(text_cols, axis=1), test_text], axis=1)
        
        self.test_case_names = test_encoded[self.case_id_col]
        self.test_X = test_X
        self.test_y = test_encoded[self.label_col]
        
        test_preproc_end_time = time.time()
        self.test_preproc_time = test_preproc_end_time - test_preproc_start_time
        
        test_start_time = time.time()
        if self.hardcoded_prediction is not None:
            # Model was trained with one class only -> probability array is always
            # 100% for the single known class, 0% for others.
            # Suppose classes_ = ['X'], we artificially create a 2D array. 
            # For demonstration, treat it as shape (n_samples, 1) or (n_samples, 2).
            # But if there was only one label, scikit-learn classifiers usually have shape (n_samples, 1).
            # We'll do a simpler approach: entire probability = 1.0 for the single class:
            predictions_proba = np.ones((test_X.shape[0], 1))
        else:
            predictions_proba = self.cls.predict_proba(test_X)
        test_end_time = time.time()
        self.test_time = test_end_time - test_start_time
        self.nr_test_cases = len(predictions_proba)
        
        return predictions_proba
