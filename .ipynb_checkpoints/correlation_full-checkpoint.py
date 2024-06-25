import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the dataset in chunks
file_path = "CSVs/sample_train.csv"
chunk_size = 10000  # Adjust chunk size as needed
chunks = pd.read_csv(file_path, chunksize=chunk_size)

# Output file path
output_file_path = "results/correlation/full_data/attribute_correlations_full.csv"

# Initialize the output file with headers
headers = ["Attribute1", "Attribute2", "Pearson_Correlation", "Spearman_Correlation", "Mutual_Information"]
with open(output_file_path, 'w') as f:
    f.write(','.join(headers) + '\n')

# Function to compute correlations and save to file
def compute_and_save_correlations(data, output_file_path):
    # Remove constant columns (features with the same value in all samples)
    data = data.loc[:, (data != data.iloc[0]).any()]
    
    # Prepare a list to store the correlation results
    correlation_results = []

    # Calculate correlations for each pair of attributes
    for attr1 in data.columns:
        for attr2 in data.columns:
            if attr1 != attr2:
                pearson_corr = data[attr1].corr(data[attr2], method="pearson")
                spearman_corr = data[attr1].corr(data[attr2], method="spearman")

                # For Mutual Information, we need to discretize continuous variables
                mutual_info = mutual_info_regression(data[[attr1]], data[attr2])[0]

                # Append the result to the list
                correlation_results.append(
                    {
                        "Attribute1": attr1,
                        "Attribute2": attr2,
                        "Pearson_Correlation": pearson_corr,
                        "Spearman_Correlation": spearman_corr,
                        "Mutual_Information": mutual_info,
                    }
                )

    # Convert the list to a DataFrame
    correlation_results_df = pd.DataFrame(correlation_results)

    # Append the results to the CSV file
    correlation_results_df.to_csv(output_file_path, mode='a', header=False, index=False)

# Process each chunk
chunk_number = 0
for chunk in chunks:
    chunk_number += 1
    logging.info(f'Processing chunk {chunk_number}')
    compute_and_save_correlations(chunk, output_file_path)

print(f"Full correlation results saved to {output_file_path}")
