import pandas as pd

# Load the dataset
file_path = "CSVs/sample_train_100.csv"
data = pd.read_csv(file_path)

# Prepare a DataFrame to store the correlation results
correlation_results = pd.DataFrame(
    columns=[
        "Attribute1",
        "Attribute2",
        "Pearson_Correlation",
        "Spearman_Correlation",
        "Mutual_Information",
    ]
)

# Calculate correlations for each pair of attributes
for i, attr1 in enumerate(data.columns):
    for attr2 in data.columns[i + 1 :]:
        pearson_corr = data[attr1].corr(data[attr2], method="pearson")
        spearman_corr = data[attr1].corr(data[attr2], method="spearman")

        # For Mutual Information, we need to discretize continuous variables
        mutual_info = mutual_info_regression(data[[attr1]], data[attr2])[0]

        # Append the result to the DataFrame
        correlation_results = correlation_results.append(
            {
                "Attribute1": attr1,
                "Attribute2": attr2,
                "Pearson_Correlation": pearson_corr,
                "Spearman_Correlation": spearman_corr,
                "Mutual_Information": mutual_info,
            },
            ignore_index=True,
        )

# Save the results to a CSV file
output_file_path = "results/correlation/attribute_correlations_full.csv"
correlation_results.to_csv(output_file_path, index=False)

print(f"Full correlation results saved to {output_file_path}")
