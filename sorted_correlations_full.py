import pandas as pd

# Load the full correlation results CSV file
correlation_results_path = "results/correlation/attribute_correlations_full.csv"
correlation_results = pd.read_csv(correlation_results_path)

# Prepare a dictionary to store sorted results for each attribute
sorted_results = {}

# List of unique attributes
attributes = correlation_results["Attribute1"].unique()

# Sort the correlations for each attribute
for attr in attributes:
    # Filter the correlations where attr is either Attribute1 or Attribute2
    filtered_results = correlation_results[
        (correlation_results["Attribute1"] == attr)
        | (correlation_results["Attribute2"] == attr)
    ].copy()

    # Replace the attribute with 'Other_Attribute' for clarity
    filtered_results["Other_Attribute"] = filtered_results.apply(
        lambda row: (
            row["Attribute2"] if row["Attribute1"] == attr else row["Attribute1"]
        ),
        axis=1,
    )

    # Drop the original Attribute1 and Attribute2 columns
    filtered_results.drop(columns=["Attribute1", "Attribute2"], inplace=True)

    # Sort by Pearson, Spearman, and Mutual Information separately and then combined
    filtered_results["Combined_Score"] = (
        filtered_results["Pearson_Correlation"].abs()
        + filtered_results["Spearman_Correlation"].abs()
        + filtered_results["Mutual_Information"]
    ) / 3

    # Sort by Combined_Score
    sorted_filtered_results = filtered_results.sort_values(
        by="Combined_Score", ascending=False
    )

    # Store the sorted results
    sorted_results[attr] = sorted_filtered_results

    # Save the sorted results to a CSV file for each attribute
    output_file_path = f"results/correlation/sorted_{attr}_correlations.csv"
    sorted_filtered_results.to_csv(output_file_path, index=False)

    print(f"Sorted correlation results for {attr} saved to {output_file_path}")

# Optionally: Save all sorted results into a single CSV file
combined_sorted_results = pd.concat(sorted_results.values(), keys=sorted_results.keys())
combined_output_file_path = "results/correlation/sorted_attribute_correlations_full.csv"
combined_sorted_results.to_csv(combined_output_file_path, index=False)
print(f"Combined sorted correlation results saved to {combined_output_file_path}")
