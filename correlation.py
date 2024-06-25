import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load the dataset
file_path = "CSVs/sample_train_100.csv"
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Calculate Pearson correlation
pearson_corr = X.corrwith(y)

# Calculate Spearman correlation
spearman_corr = X.corrwith(y, method="spearman")

# Calculate Mutual Information
mutual_info = mutual_info_classif(X, y, discrete_features=False)

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
