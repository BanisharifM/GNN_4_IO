import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Load the dataset
file_path = "CSVs/sample_train_100.csv"
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Remove constant columns (features with the same value in all samples)
X = X.loc[:, (X != X.iloc[0]).any()]

# Calculate Pearson correlation, handling NaN values
pearson_corr = X.corrwith(y, method="pearson").fillna(0)

# Calculate Spearman correlation, handling NaN values
spearman_corr = X.corrwith(y, method="spearman").fillna(0)

# Ensure the target variable is categorical for mutual information
if y.dtype != "object":
    y = y.astype("category")

# Calculate Mutual Information
mutual_info = mutual_info_classif(X, y, discrete_features="auto")

# Create a DataFrame to hold the results
correlation_results = pd.DataFrame(
    {
        "Attribute": X.columns,
        "Pearson_Correlation": pearson_corr.values,
        "Spearman_Correlation": spearman_corr.values,
        "Mutual_Information": mutual_info,
    }
)

# Save the results to a new CSV file
output_file_path = "results/correlation/attribute_correlations.csv"
correlation_results.to_csv(output_file_path, index=False)

print(f"Correlation results saved to {output_file_path}")
