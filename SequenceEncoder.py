import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

class SequenceEncoder():
    
    def __init__(self, nr_events, event_nr_col, case_id_col, label_col, 
                 static_cols=None, dynamic_cols=None, cat_cols=None, 
                 event_col=None, encoding_method="label", oversample_fit=True, 
                 minority_label="0", fillna=True, random_state=22):
        
        self.nr_events = nr_events
        self.event_nr_col = event_nr_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.static_cols = static_cols if static_cols else []
        self.dynamic_cols = dynamic_cols if dynamic_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.event_col = event_col
        self.encoding_method = encoding_method
        self.oversample_fit = oversample_fit
        self.minority_label = minority_label
        self.fillna = fillna
        self.random_state = random_state
        self.vectorizer = None
        self.fitted_features = None
        self.is_fitted = False
        self.encoders = {}  # Dictionary to store encoders for each categorical column
        self.event_encoders = {}  # Separate dictionary for event encoders

    def _complex_encode(self, X, fitting=False):
        # Group rare LoanGoal categories
        if 'LoanGoal' in X.columns:
            # Create a mapping for rare categories (less than 1% of data)
            if fitting:
                total_cases = len(X[self.case_id_col].unique())
                category_counts = X.groupby(self.case_id_col)['LoanGoal'].first().value_counts()
                rare_categories = category_counts[category_counts/total_cases < 0.01].index
                
                self.loan_goal_mapping = {}
                for cat in rare_categories:
                    if 'business' in cat.lower():
                        self.loan_goal_mapping[cat] = 'Business'
                    elif any(x in cat.lower() for x in ['boat', 'motorcycle', 'caravan']):
                        self.loan_goal_mapping[cat] = 'Vehicle Other'
                    elif 'debt' in cat.lower() or 'restructuring' in cat.lower():
                        self.loan_goal_mapping[cat] = 'Debt Management'
                    elif 'tax' in cat.lower():
                        self.loan_goal_mapping[cat] = 'Tax and Payments'
                    else:
                        self.loan_goal_mapping[cat] = 'Other'
            
            # Apply the mapping
            if hasattr(self, 'loan_goal_mapping'):
                X['LoanGoal'] = X['LoanGoal'].replace(self.loan_goal_mapping)
        
        # Rest of the encoding process
        max_events = min(self.nr_events, X[self.event_nr_col].max())
        
        # Get all first events for each case
        first_events = X.groupby(self.case_id_col).first().reset_index()
        
        data_final = first_events[self.static_cols].copy()
        data_final[self.case_id_col] = first_events[self.case_id_col]
        data_final[self.label_col] = first_events[self.label_col]
        
        if self.encoding_method == "onehot":
            # Process each event up to max_events
            for event_nr in range(1, max_events + 1):
                # Get events for current event number
                event_data = X[X[self.event_nr_col] == event_nr].copy()
                
                if len(event_data) == 0:
                    continue
                
                # First encode the event name if specified
                if self.event_col and self.event_col in X.columns:
                    event_name_data = event_data[self.event_col].fillna('MISSING')
                    
                    if fitting:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded = encoder.fit_transform(event_name_data.values.reshape(-1, 1))
                        self.event_encoders[f"event{event_nr}"] = encoder
                    else:
                        encoder_key = f"event{event_nr}"
                        if encoder_key not in self.event_encoders:
                            raise ValueError(f"No encoder found for {encoder_key}")
                        encoded = self.event_encoders[encoder_key].transform(event_name_data.values.reshape(-1, 1))
                    
                    # Create column names for the encoded event names
                    feature_names = [f"event{event_nr}_{val}" for val in self.event_encoders[f"event{event_nr}"].categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=event_data.index)
                    
                    # Join with case IDs to ensure correct alignment
                    encoded_df[self.case_id_col] = event_data[self.case_id_col]
                    data_final = data_final.merge(encoded_df, on=self.case_id_col, how='left')
                
                # Then process each categorical column
                if self.cat_cols:
                    for col in self.cat_cols:
                        if col in X.columns:
                            # Get data for current event
                            col_data = event_data[col].fillna('MISSING')
                            
                            if fitting:
                                # Create and fit a new encoder for this column and event
                                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                                encoded = encoder.fit_transform(col_data.values.reshape(-1, 1))
                                self.encoders[f"{col}_event{event_nr}"] = encoder
                            else:
                                # Use the previously fitted encoder
                                encoder_key = f"{col}_event{event_nr}"
                                if encoder_key not in self.encoders:
                                    raise ValueError(f"No encoder found for {encoder_key}")
                                encoded = self.encoders[encoder_key].transform(col_data.values.reshape(-1, 1))
                            
                            # Create column names for the encoded features
                            feature_names = [f"{col}_event{event_nr}_{val}" for val in self.encoders[f"{col}_event{event_nr}"].categories_[0]]
                            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=event_data.index)
                            
                            # Join with case IDs to ensure correct alignment
                            encoded_df[self.case_id_col] = event_data[self.case_id_col]
                            data_final = data_final.merge(encoded_df, on=self.case_id_col, how='left')
                
                # Add dynamic columns if any
                if self.dynamic_cols:
                    dynamic_cols_event = [f"{col}_event{event_nr}" for col in self.dynamic_cols]
                    event_dynamic_data = event_data[self.dynamic_cols].copy()
                    event_dynamic_data.columns = dynamic_cols_event
                    
                    # Join with case IDs to ensure correct alignment
                    event_dynamic_data[self.case_id_col] = event_data[self.case_id_col]
                    data_final = data_final.merge(event_dynamic_data, on=self.case_id_col, how='left')
            
            # Set is_fitted to True after all encodings are done
            if fitting:
                self.is_fitted = True
        
        # Fill NaN values with 0 for encoded columns
        data_final = data_final.fillna(0)
        
        return data_final

    def fit_transform(self, X):
        return self._encode(X, fitting=True)

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Call fit_transform before transform")
        return self._encode(X, fitting=False)

    def _encode(self, X, fitting=False):
        return self._complex_encode(X, fitting)
