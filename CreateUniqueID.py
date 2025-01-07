import hashlib
import pandas as pd

# Load the data
file_path = 'data/BPI_Challenge_2017_unstructured.csv'  # Replace with your file path
data = pd.read_csv(file_path, delimiter=';')

# Create a new column for the unique event ID using incremental integers
data['UniqueEventID'] = range(1, len(data) + 1)

# Save the updated dataset
output_file_path = 'data/BPI_Challenge_2017_unstructured_unique.csv'
data.to_csv(output_file_path, index=False, sep=';')

print(f"Unique Event ID column added. Updated file saved at: {output_file_path}")
