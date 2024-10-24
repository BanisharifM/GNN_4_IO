import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_data(file_path, total_rows=None, output_dir='NN/NN_101', test_size=0.25):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    logging.info(f"Loading dataset from {file_path}")
    data = pd.read_csv(file_path)
    logging.info(f"Dataset loaded successfully with {len(data)} rows.")

    # Select specific number of rows, or use all rows if total_rows is None
    if total_rows is not None and len(data) > total_rows:
        data = data.sample(n=total_rows, random_state=1).reset_index(drop=True)
        logging.info(f"Selected {total_rows} rows from the dataset.")
    else:
        logging.info(f"Processing all {len(data)} rows.")

    # Split dataset into training and testing sets
    logging.info(f"Splitting dataset into training and testing sets with test size = {test_size}")
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=1)
    logging.info(f"Dataset split into {len(train_data)} training rows and {len(test_data)} testing rows.")

    # Save training data
    train_file = os.path.join(output_dir, 'train_data.csv')
    train_data.to_csv(train_file, index=False)
    logging.info(f"Training data saved to {train_file}.")

    # Save testing data
    test_file = os.path.join(output_dir, 'test_data.csv')
    test_data.to_csv(test_file, index=False)
    logging.info(f"Testing data saved to {test_file}.")

# Define paths and parameters
file_path = 'CSVs/sample_train_total.csv'
total_rows = None  # Set to None to process all rows
output_dir = 'NN/NN_103'

# Process the data
logging.info("Starting data processing.")
load_and_process_data(file_path, total_rows, output_dir)
logging.info("Data processing completed.")
