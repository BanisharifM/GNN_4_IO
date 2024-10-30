import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the correlation results CSV file
correlation_results_path = "results/correlation/attribute_correlations.csv"
correlation_results = pd.read_csv(correlation_results_path)

# Normalize the correlation values (optional)
scaler = MinMaxScaler()
correlation_results[
    ["Pearson_Correlation", "Spearman_Correlation", "Mutual_Information"]
] = scaler.fit_transform(
    correlation_results[
        ["Pearson_Correlation", "Spearman_Correlation", "Mutual_Information"]
    ]
)

# Calculate a combined score
correlation_results["Combined_Score"] = (
    correlation_results["Pearson_Correlation"]
    + correlation_results["Spearman_Correlation"]
    + correlation_results["Mutual_Information"]
) / 3

# Sort the attributes based on the combined score
sorted_results = correlation_results.sort_values(by="Combined_Score", ascending=False)

# Save the sorted results to a new CSV file
output_file_path = "results/correlation/sorted_attribute_correlations.csv"
sorted_results.to_csv(output_file_path, index=False)

print(f"Sorted correlation results saved to {output_file_path}")
