import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV

class SequenceEncoder():
    
    def __init__(self, nr_events, event_nr_col, case_id_col, label_col, 
                 static_cols=None, dynamic_cols=None, last_state_cols=None,
                 cat_cols=None, oversample_fit=True, minority_label="positive", 
                 fillna=True, random_state=None, max_events=200, 
                 dyn_event_marker="dynevent", last_event_marker="lastevent",
                 case_length_col="case_length", pre_encoded=False):
        
        # Use lists instead of None for mutable defaults
        self.nr_events = nr_events
        self.event_nr_col = event_nr_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.static_cols = static_cols if static_cols else []
        self.dynamic_cols = dynamic_cols if dynamic_cols else []
        self.last_state_cols = last_state_cols if last_state_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        
        self.oversample_fit = oversample_fit
        self.minority_label = minority_label
        self.fillna = fillna
        self.random_state = random_state
        
        self.max_events = max_events
        self.dyn_event_marker = dyn_event_marker
        self.last_event_marker = last_event_marker
        self.case_length_col = case_length_col
        self.pre_encoded = pre_encoded
        
        self.fitted_columns = None
        
    def fit(self, X):
        """No actual fitting of encoders beyond columns is done here.
        Will store the final columns after transforming the train set 
        so we can align columns for the test set."""
        return self
        
    def fit_transform(self, X):
        """Encode the data and oversample if required."""
        data = self._encode(X)
        if self.oversample_fit:
            data = self._oversample(data)
        return data
        
    def transform(self, X):
        """Encode the data without oversampling (for test or unseen data)."""
        data = self._encode(X)
        return data
    
    def pre_encode(self, X):
        """
        An alternative pre-encoding routine that constructs all potential columns
        up to `max_events` events, then merges them. If used, you can set
        `pre_encoded=True` to avoid repeated encoding.
        """
        # Ensure label_col and case_id_col are in static_cols
        if self.label_col not in self.static_cols:
            self.static_cols.append(self.label_col)
        if self.case_id_col not in self.static_cols:
            self.static_cols.append(self.case_id_col)
        
        # Start with the rows for the first event only (for static attributes)
        data_final = X[X[self.event_nr_col] == 1][self.static_cols]

        # Merge in dynamic columns for all possible events up to `max_events`
        for i in range(1, self.max_events + 1):
            data_selected = X[X[self.event_nr_col] == i][[self.case_id_col] + self.dynamic_cols]
            data_selected.columns = (
                [self.case_id_col] 
                + [f"{col}_{self.dyn_event_marker}{i}" for col in self.dynamic_cols]
            )
            data_final = pd.merge(data_final, data_selected, on=self.case_id_col, how="left")
         
        # Merge in last-state columns for all events, forward-filling as needed
        for i in range(1, self.max_events + 1):
            data_selected = X[X[self.event_nr_col] == i][[self.case_id_col] + self.last_state_cols]
            data_selected.columns = (
                [self.case_id_col] 
                + [f"{col}_{self.last_event_marker}{i}" for col in self.last_state_cols]
            )
            data_final = pd.merge(data_final, data_selected, on=self.case_id_col, how="left")
            
            # Forward-fill each last_state_col from the previous event
            if i > 1:
                for col in self.last_state_cols:
                    col_name = f"{col}_{self.last_event_marker}{i}"
                    prev_col_name = f"{col}_{self.last_event_marker}{i-1}"
                    missing = data_final[col_name].isnull()
                    data_final.loc[missing, col_name] = data_final.loc[missing, prev_col_name]
                    
        # Handle categorical columns via DictVectorizer
        dynamic_cat_cols = [col for col in self.cat_cols if col in self.dynamic_cols]
        static_cat_cols = [col for col in self.cat_cols if col in self.static_cols]
        categorical_cols = [
            f"{col}_{self.dyn_event_marker}{i}" for i in range(1, self.max_events + 1) 
            for col in dynamic_cat_cols
        ] + static_cat_cols
        
        cat_df = data_final[categorical_cols]
        cat_dict = cat_df.to_dict(orient='records')  # changed .T.to_dict().values() to simpler approach
        vectorizer = DV(sparse=False)
        vec_cat = vectorizer.fit_transform(cat_dict)
        cat_data = pd.DataFrame(vec_cat, columns=vectorizer.feature_names_)
        
        # Drop old categorical cols, then concatenate
        data_final.drop(columns=categorical_cols, inplace=True)
        data_final = pd.concat([data_final, cat_data], axis=1)
        
        # Add case_length (number of events in each case)
        # Updated to non-deprecated syntax
        length_df = (
            X.groupby(self.case_id_col)[self.event_nr_col]
             .agg(case_length="max")
             .reset_index()
        )
        data_final = pd.merge(data_final, length_df, on=self.case_id_col, how="left")
    
        # Fill missing values
        if self.fillna:
            for col in data_final:
                dt = data_final[col].dtype 
                if dt == int or dt == float:
                    data_final[col].fillna(0, inplace=True)
                else:
                    data_final[col].fillna("", inplace=True)

        return data_final

    def _encode(self, X):
        """
        Main entry for encoding. If `pre_encoded=True`, we assume the data 
        has already been processed by `pre_encode`. Otherwise, we call
        `_complex_encode`.
        """
        if self.pre_encoded:
            # Filter out columns for events beyond nr_events
            # and keep only rows where the case length >= nr_events
            # so we discard cases shorter than needed
            cols_to_drop = []
            
            # We want to drop columns that contain dyn_event_marker + 
            # the event index if that index is > self.nr_events
            for i in range(self.nr_events + 1, self.max_events + 1):
                marker = f"{self.dyn_event_marker}{i}"
                cols_to_drop.append(marker)
            
            # Also drop columns for last_event_marker if event index > nr_events
            # or possibly keep them all if you want them
            # For simplicity, let's keep only up to the `nr_events` if needed
            # (But in practice, you might want all last-event columns.)
            for i in range(self.nr_events + 1, self.max_events + 1):
                marker = f"{self.last_event_marker}{i}"
                cols_to_drop.append(marker)
            
            # Actually filter columns
            keep_cols = []
            for c in X.columns:
                # Keep columns that do not contain these marker strings
                if not any(m in c for m in cols_to_drop):
                    keep_cols.append(c)
            
            # Filter the data
            selected = X[keep_cols].copy()
            
            # Keep only rows with enough events
            selected = selected[selected[self.case_length_col] >= self.nr_events]
            
            # Drop the case_length_col if you don't want it as a feature
            selected.drop(self.case_length_col, axis=1, inplace=True)
            return selected.reset_index(drop=True)
        
        else:
            return self._complex_encode(X)
        
    def _complex_encode(self, X):
        """
        Encodes up to `nr_events` events by merging dynamic columns for events 1..nr_events,
        merges last-state columns from the nth event, then one-hot encodes the categorical cols.
        """
        
        # Ensure label_col and case_id_col in static_cols
        if self.label_col not in self.static_cols:
            self.static_cols.append(self.label_col)
        if self.case_id_col not in self.static_cols:
            self.static_cols.append(self.case_id_col)
        
        # Start with the rows for the first event only (for static attributes)
        data_final = X[X[self.event_nr_col] == 1][self.static_cols].copy()
        
        # Merge dynamic columns for event i=1..nr_events
        # Typically, 'how="left"' is safer so we do not lose the original row set
        for i in range(1, self.nr_events + 1):
            data_selected = X[X[self.event_nr_col] == i][[self.case_id_col] + self.dynamic_cols].copy()
            # Rename dynamic columns to col_i
            data_selected.columns = [self.case_id_col] + [f"{col}_{i}" for col in self.dynamic_cols]
            data_final = pd.merge(data_final, data_selected, on=self.case_id_col, how="left")
        
        # Merge in last-state columns from the nth event only
        # Then forward-fill from the previous event if missing
        if self.last_state_cols:
            last_event_df = X[X[self.event_nr_col] == self.nr_events][[self.case_id_col] + self.last_state_cols].copy()
            data_final = pd.merge(data_final, last_event_df, on=self.case_id_col, how="left")
            
            # Forward-fill logic if the row from the nth event was empty 
            # (We attempt to look back to event nr_events-1, then nr_events-2, etc.)
            # This approach can be heavy if the dataset is large. Use carefully.
            for idx, row in data_final.iterrows():
                # If any last_state_col is NaN, look backward
                for col in self.last_state_cols:
                    if pd.isnull(row[col]):
                        current_nr_events = self.nr_events - 1
                        while pd.isnull(row[col]) and current_nr_events > 0:
                            prev_row = X[
                                (X[self.case_id_col] == row[self.case_id_col]) 
                                & (X[self.event_nr_col] == current_nr_events)
                            ]
                            # If prev_row is not empty, fill from there
                            if not prev_row.empty:
                                row[col] = prev_row.iloc[0][col]
                            current_nr_events -= 1
                data_final.loc[idx] = row
        
        # Identify categorical columns
        dynamic_cat_cols = [c for c in self.cat_cols if c in self.dynamic_cols]
        static_cat_cols = [c for c in self.cat_cols if c in self.static_cols]
        
        # Combine all columns that need one-hot encoding
        # For dynamic, we have them repeated per event (col_i).
        categorical_cols = (
            [f"{col}_{i}" for i in range(1, self.nr_events + 1) for col in dynamic_cat_cols]
            + static_cat_cols
        )
        
        cat_df = data_final[categorical_cols].copy()
        
        # Convert categorical data into a list-of-dict for DictVectorizer
        cat_dict = cat_df.to_dict(orient='records')
        
        vectorizer = DV(sparse=False)
        vec_cat = vectorizer.fit_transform(cat_dict)
        cat_data = pd.DataFrame(vec_cat, columns=vectorizer.feature_names_)
        
        # Drop the original categorical cols, then concatenate the new one-hot columns
        data_final.drop(columns=categorical_cols, inplace=True)
        data_final = pd.concat([data_final, cat_data], axis=1)
        
        # Align columns to ensure consistency between fit and transform
        if self.fitted_columns is not None:
            # If there are columns we saw in training but don't see now, create them with 0
            missing_cols = self.fitted_columns[~self.fitted_columns.isin(data_final.columns)]
            for col in missing_cols:
                data_final[col] = 0
            # If there are new columns not in fitted_columns, drop them
            extra_cols = [c for c in data_final.columns if c not in self.fitted_columns]
            if extra_cols:
                data_final.drop(columns=extra_cols, inplace=True)
            
            # Reorder columns to match the training set
            data_final = data_final[self.fitted_columns]
        else:
            # Store these columns for future transforms
            self.fitted_columns = data_final.columns
        
        # Fill missing values
        if self.fillna:
            for col in data_final:
                dt = data_final[col].dtype 
                if dt == int or dt == float:
                    data_final[col].fillna(0, inplace=True)
                else:
                    data_final[col].fillna("", inplace=True)
                    
        # Reset index for cleanliness
        data_final.reset_index(drop=True, inplace=True)
        
        return data_final
    
    def _oversample(self, X):
        """
        Simple oversampling approach: 
        - Calculate how many times the minority label is needed to match majority. 
        - Randomly sample from the minority label rows with replacement.
        """
        # Count how many more samples we need to match majority
        minority_count = sum(X[self.label_col] == self.minority_label)
        majority_count = len(X) - minority_count
        oversample_count = majority_count - minority_count
        
        # If oversample_count > 0, it means minority is smaller than majority
        if oversample_count > 0 and minority_count > 0:
            oversampled_data = X[X[self.label_col] == self.minority_label].sample(
                oversample_count, 
                replace=True, 
                random_state=self.random_state
            )
            X = pd.concat([X, oversampled_data], ignore_index=True)
        
        return X
