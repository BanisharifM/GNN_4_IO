import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the dataset in chunks
file_path = "CSVs/sample_train.csv"
chunk_size = 10000  # Adjust chunk size as needed
chunks = pd.read_csv(file_path, chunksize=chunk_size)

# Output file path
output_file_path = "results/correlation/full_data/attribute_correlations_full.csv"

# Initialize the output file with headers if it does not exist
if not os.path.exists(output_file_path):
    headers = ["Attribute1", "Attribute2", "Pearson_Correlation", "Spearman_Correlation", "Mutual_Information"]
    with open(output_file_path, 'w') as f:
        f.write(','.join(headers) + '\n')

# Function to compute correlations
def compute_correlations(data):
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
    return correlation_results_df

# Process each chunk
chunk_number = 0
for chunk in chunks:
    chunk_number += 1
    logging.info(f'Processing chunk {chunk_number}')
    new_correlation_results_df = compute_correlations(chunk)
    
    # Read the existing results if the file is not empty
    if os.path.getsize(output_file_path) > 0:
        existing_correlation_results_df = pd.read_csv(output_file_path)
        # Combine the new results with the existing ones
        combined_results_df = pd.concat([existing_correlation_results_df, new_correlation_results_df], ignore_index=True)
    else:
        combined_results_df = new_correlation_results_df
    
    # Save the combined results back to the CSV file
    combined_results_df.to_csv(output_file_path, index=False)

print(f"Full correlation results saved to {output_file_path}")
