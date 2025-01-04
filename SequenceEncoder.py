import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

class SequenceEncoder():
    
    def __init__(self, nr_events, event_nr_col, case_id_col, label_col, 
                 static_cols=None, dynamic_cols=None, cat_cols=None, 
                 encoding_method="label", oversample_fit=True, 
                 minority_label="0", fillna=True, random_state=22):
        
        self.nr_events = nr_events
        self.event_nr_col = event_nr_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.static_cols = static_cols if static_cols else []
        self.dynamic_cols = dynamic_cols if dynamic_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.encoding_method = encoding_method
        self.oversample_fit = oversample_fit
        self.minority_label = minority_label
        self.fillna = fillna
        self.random_state = random_state
        self.vectorizer = None
        self.fitted_features = None
        self.is_fitted = False  # New flag added

    def _complex_encode(self, X, fitting=False):
        print("Input shape:", X.shape)
        print("Available columns:", X.columns.tolist())
        
        # Store original case_id and label columns
        case_ids = X[X[self.event_nr_col] == 1][self.case_id_col].copy()
        labels = X[X[self.event_nr_col] == 1][self.label_col].copy()
        
        # Get static features from first events
        data_final = X[X[self.event_nr_col] == 1][self.static_cols].copy()
        print("Static data shape:", data_final.shape)
        
        # Create dictionary for categorical variables
        cat_dict = []
        if self.cat_cols:
            for idx in data_final.index:
                case_dict = {}
                for col in self.cat_cols:
                    if col in X.columns:
                        val = str(X.loc[idx, col])
                        case_dict[f"{col}_{val}"] = 1
                if case_dict:
                    cat_dict.append(case_dict)
        
        print("Number of categorical dictionaries:", len(cat_dict))
        
        if cat_dict:
            if fitting:
                print("Fitting vectorizer...")
                self.vectorizer = DictVectorizer(sparse=False)
                vec_cat = self.vectorizer.fit_transform(cat_dict)
                self.fitted_features = self.vectorizer.get_feature_names_out()
                self.is_fitted = True
                print(f"Fitted features count: {len(self.fitted_features)}")
            else:
                if not self.is_fitted:
                    raise ValueError("Encoder has not been fitted yet.")
                print("Transforming with fitted vectorizer...")
                vec_cat = self.vectorizer.transform(cat_dict)
            
            data_cat = pd.DataFrame(vec_cat, 
                                  columns=self.fitted_features, 
                                  index=data_final.index)
            data_cat = data_cat.fillna(0)
            data_final = pd.concat([data_final, data_cat], axis=1)
        
        # Add back case_id and label columns
        data_final[self.case_id_col] = case_ids
        data_final[self.label_col] = labels
        
        return data_final

    def fit_transform(self, X):
        return self._encode(X, fitting=True)

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Call fit_transform before transform")
        return self._encode(X, fitting=False)

    def _encode(self, X, fitting=False):
        return self._complex_encode(X, fitting)
