import pandas as pd
import numpy as np

# Load the mutual information matrix
mi_matrix_path = 'Correlation/2/mutual_information2.csv'  # Adjust the path as necessary
mi_matrix = pd.read_csv(mi_matrix_path, index_col=0)

# Flatten the matrix to get mutual information values
mi_values = mi_matrix.values.flatten()
mi_values = mi_values[~np.isnan(mi_values)]  # Remove NaN values

# Calculate mean and standard deviation
mean_mi = np.mean(mi_values)
std_mi = np.std(mi_values)

# Calculate Z > 2 threshold
z_threshold_2 = mean_mi + 2 * std_mi

print(f'Mean MI: {mean_mi}')
print(f'Standard Deviation of MI: {std_mi}')
print(f'Z > 2 Threshold: {z_threshold_2}')

# ---------------------------------------------
# Mean MI: 0.16363906899627848
# Standard Deviation of MI: 0.17290683961371273
# Z > 2 Threshold: 0.5094527482237039
# ---------------------------------------------


# Load the mutual information matrix
mi_matrix_path = 'Correlation/2/mutual_information2.csv'  # Adjust the path as necessary
mi_matrix = pd.read_csv(mi_matrix_path, index_col=0)

# Apply the threshold of 0.80 to determine the edges
threshold_80 = 0.50
edges_80 = (mi_matrix.values > threshold_80).sum() - len(mi_matrix)+1  # Exclude diagonal elements
edges_80 = edges_80 / 2  # Each edge is counted twice, so divide by 2

# Flatten the matrix to get mutual information values
mi_values = mi_matrix.values.flatten()
mi_values = mi_values[~np.isnan(mi_values)]  # Remove NaN values

# Calculate the 90th percentile threshold
percentile_90 = np.percentile(mi_values, 90)
edges_90th_percentile = (mi_matrix.values > percentile_90).sum() - len(mi_matrix)  # Exclude diagonal elements
edges_90th_percentile = edges_90th_percentile / 2  # Each edge is counted twice, so divide by 2

print(f'Number of edges with threshold {threshold_80}: {int(edges_80)}')
print(f'90th Percentile threshold: {percentile_90}')
print(f'Number of edges with 90th Percentile threshold: {int(edges_90th_percentile)}')

# ---------------------------------------------
# Number of edges with threshold 0.5: 18
# 90th Percentile threshold: 0.3259455280319236
# Number of edges with 90th Percentile threshold: 82
# ---------------------------------------------


import pandas as pd
import numpy as np

def generate_graph_edges(mi_matrix_path, output_path, threshold):
    # Load the mutual information matrix
    mi_matrix = pd.read_csv(mi_matrix_path, index_col=0)
    
    # Find the edges that meet the threshold
    edges = []
    for i in range(mi_matrix.shape[0]):
        for j in range(i + 1, mi_matrix.shape[1]):
            if mi_matrix.iloc[i, j] > threshold:
                edges.append((mi_matrix.index[i], mi_matrix.columns[j], mi_matrix.iloc[i, j]))
    
    # Convert edges to a DataFrame
    edges_df = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
    
    # Save the edges to a CSV file
    edges_df.to_csv(output_path, index=False)
    print(f'Edges with threshold {threshold} saved to {output_path}')

# Define the paths and threshold
mi_matrix_path = 'Correlation/2/mutual_information2.csv'  # Adjust the path as necessary
output_path = 'Correlation/2/graph_edges.csv'  # Adjust the output path as necessary
threshold = 0.3259455280319236  # Set your dynamic threshold here

# Generate graph edges
generate_graph_edges(mi_matrix_path, output_path, threshold)

# ---------------------------------------------
# Edges with threshold 0.3259455280319236 saved to Correlation/2/graph_edges.csv
# ---------------------------------------------