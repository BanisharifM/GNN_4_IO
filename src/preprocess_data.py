import pandas as pd
from sklearn.model_selection import train_test_split

def process_full_data(file_path):
    # Load the entire dataset
    data = pd.read_csv(file_path)

    # Shuffle the dataset thoroughly
    data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data_shuffled, test_size=0.2, random_state=42)
    
    # Save the split data to new CSV files
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

# Example usage:
file_path = 'CSVs/sample_train.csv'
process_full_data(file_path)
