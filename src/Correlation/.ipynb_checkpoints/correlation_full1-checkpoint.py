import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# Load the dataset
file_path = "CSVs/sample_train.csv"
data = pd.read_csv(file_path)

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

# Save the results to a CSV file
output_file_path = "results/correlation/full_data/attribute_correlations_full.csv"
correlation_results_df.to_csv(output_file_path, index=False)

print(f"Full correlation results saved to {output_file_path}")
