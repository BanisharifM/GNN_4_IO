import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

# Function to compute mutual information for each pair of columns
def compute_mutual_information(data):
    columns = data.columns
    n = len(columns)
    mi_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            mi_matrix[i, j] = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
    
    return pd.DataFrame(mi_matrix, index=columns, columns=columns)

# File paths
file_path = 'CSVs/sample_train.csv'
output_path = 'Correlation/mutual_information.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Calculate the mutual information matrix
mi_matrix = compute_mutual_information(data)

# Save the mutual information matrix to a new CSV file
mi_matrix.to_csv(output_path)

print(f'Mutual information matrix saved to {output_path}')
