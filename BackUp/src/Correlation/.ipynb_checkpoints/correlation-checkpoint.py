import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

# Load the dataset
file_path = "CSVs/sample_train_100.csv"
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop(columns=["tag"])
y = data["tag"]

# Print initial columns
print("Initial columns:", X.columns.tolist())

# Remove constant columns (features with the same value in all samples)
non_constant_columns = X.loc[:, (X != X.iloc[0]).any()]
print("Columns after removing constants:", non_constant_columns.columns.tolist())

# Calculate Pearson correlation, handling NaN values
pearson_corr = non_constant_columns.corrwith(y, method="pearson").fillna(0)

# Calculate Spearman correlation, handling NaN values
spearman_corr = non_constant_columns.corrwith(y, method="spearman").fillna(0)

# Calculate Mutual Information for regression
mutual_info = mutual_info_regression(non_constant_columns, y, discrete_features="auto")

# Create a DataFrame to hold the results
correlation_results = pd.DataFrame(
    {
        "Attribute": non_constant_columns.columns,
        "Pearson_Correlation": pearson_corr.values,
        "Spearman_Correlation": spearman_corr.values,
        "Mutual_Information": mutual_info,
    }
)

# Save the results to a new CSV file
output_file_path = "results/correlation/attribute_correlations.csv"
correlation_results.to_csv(output_file_path, index=False)

print(f"Correlation results saved to {output_file_path}")
