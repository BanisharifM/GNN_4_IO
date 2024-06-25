import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data to get the list of attributes
data = pd.read_csv('CSVs/sample_train_100.csv')
data = data.loc[:, (data != data.iloc[0]).any()]
attributes = data.columns

# Ensure the output directory exists
visualization_dir = "results/correlation/visualization"
os.makedirs(visualization_dir, exist_ok=True)

# Function to create and save visualizations
def create_visualizations(attribute, sorted_results):
    # Bar chart of combined scores
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Combined_Score", y="attr2", data=sorted_results)
    plt.title(f"Combined Correlation Scores for {attribute}")
    plt.xlabel("Combined Score")
    plt.ylabel("Attribute")
    plt.tight_layout()
    bar_chart_path = f"{visualization_dir}/combined_scores_bar_chart_{attribute}.png"
    plt.savefig(bar_chart_path)
    plt.close()

    # Heatmap of Pearson and Spearman correlations
    correlation_matrix = sorted_results[
        ["attr2", "Pearson_Correlation", "Spearman_Correlation"]
    ].set_index("attr2")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Heatmap of Pearson and Spearman Correlations for {attribute}")
    plt.tight_layout()
    heatmap_path = f"{visualization_dir}/correlation_heatmap_{attribute}.png"
    plt.savefig(heatmap_path)
    plt.close()

# Generate visualizations for each attribute
for attr in attributes:
    sorted_results_path = f"results/correlation/sorted/sorted_{attr}_correlations.xlsx"
    sorted_results = pd.read_excel(sorted_results_path)
    create_visualizations(attr, sorted_results)

print("Visualizations created and saved.")

