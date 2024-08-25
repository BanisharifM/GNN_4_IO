import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sorted correlation results CSV file
sorted_results_path = "results/correlation/sorted_attribute_correlations.csv"
sorted_results = pd.read_csv(sorted_results_path)

# Bar chart of combined scores
plt.figure(figsize=(10, 8))
sns.barplot(x="Combined_Score", y="Attribute", data=sorted_results)
plt.title("Combined Correlation Scores for Attributes")
plt.xlabel("Combined Score")
plt.ylabel("Attribute")
plt.tight_layout()
plt.savefig("results/correlation/combined_scores_bar_chart.png")
plt.show()

# Heatmap of Pearson and Spearman correlations
correlation_matrix = sorted_results[
    ["Attribute", "Pearson_Correlation", "Spearman_Correlation"]
].set_index("Attribute")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Heatmap of Pearson and Spearman Correlations")
plt.tight_layout()
plt.savefig("results/correlation/correlation_heatmap.png")
plt.show()

# Pairplot for top 5 attributes (based on combined score)
top_attributes = sorted_results["Attribute"].head(5)
top_data = pd.read_csv("CSVs/sample_train_100.csv")[top_attributes.to_list() + ["tag"]]
sns.pairplot(top_data)
plt.suptitle("Pairplot for Top 5 Attributes and Target", y=1.02)
plt.tight_layout()
plt.savefig("results/correlation/top_attributes_pairplot.png")
plt.show()
