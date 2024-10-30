import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score

# Function to compute mutual information for each pair of columns and normalize it
def compute_normalized_mutual_information(data, n_bins=10):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    columns = data.columns
    n = len(columns)
    mi_matrix = np.zeros((n, n))
    
    # Discretize continuous data
    discretized_data = discretizer.fit_transform(data)
    
    for i in range(n):
        for j in range(n):
            mi = mutual_info_score(discretized_data[:, i], discretized_data[:, j])
            h_i = mutual_info_score(discretized_data[:, i], discretized_data[:, i])
            h_j = mutual_info_score(discretized_data[:, j], discretized_data[:, j])
            if h_i > 0 and h_j > 0:
                mi_matrix[i, j] = mi / np.sqrt(h_i * h_j)
            else:
                mi_matrix[i, j] = 0
    
    return pd.DataFrame(mi_matrix, index=columns, columns=columns)

# File paths
file_path = 'CSVs/sample_train_100.csv'
output_path = 'Correlation/mutual_information2_100.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Calculate the normalized mutual information matrix
mi_matrix = compute_normalized_mutual_information(data)

# Save the normalized mutual information matrix to a new CSV file
mi_matrix.to_csv(output_path)

print(f'Normalized mutual information matrix saved to {output_path}')
