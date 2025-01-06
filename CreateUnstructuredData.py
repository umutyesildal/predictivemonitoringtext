import pandas as pd
import random

# Load the dataset
file_path = 'data/BPI_Challenge_2017.csv'
data = pd.read_csv(file_path, delimiter=';')

# Function to generate random event descriptions based on row data
def generate_event_description(row):
    event = row['event']
    application_type = row['ApplicationType']
    loan_goal = row['LoanGoal']
    requested_amount = row['RequestedAmount']
    event_origin = row['EventOrigin']

    descriptions = [
        f"{event} for a {application_type} application.",
        f"Loan goal: {loan_goal}. Processing event: {event}.",
        f"An application for {loan_goal} was {event_origin}.",
        f"Requested amount of {requested_amount} is under review.",
        f"The event '{event}' occurred for a case related to {application_type}.",
        f"Details processed for loan amount: {requested_amount}.",
        f"{event} initiated for a {application_type} with goal: {loan_goal}.",
        "Action recorded during application review.",
        "Application event is being tracked for compliance.",
        f"Loan application event: {event_origin}."
    ]

    return random.choice(descriptions)

# Add the new EventDescription column
data['EventDescription'] = data.apply(generate_event_description, axis=1)

# Save the updated dataset with the new column
output_path = 'data/BPI_Challenge_2017_unstructured.csv'
data.to_csv(output_path, index=False, sep=';')

print(f"Updated dataset saved to: {output_path}")