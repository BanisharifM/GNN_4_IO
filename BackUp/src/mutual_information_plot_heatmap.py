import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the mutual information values csv file
file_path = 'Correlation/2/mutual_information2_100.csv'
mutual_info_data = pd.read_csv(file_path, index_col=0)

# Subtract 2 from all mutual information values
adjusted_mutual_info_data = mutual_info_data - 0

# Generate a heatmap plot for each feature
for feature in adjusted_mutual_info_data.columns:
    sorted_data = adjusted_mutual_info_data[[feature]].sort_values(by=feature, ascending=False)
    plt.figure(figsize=(10, 12))  # Increase the figure size for better readability
    sns.heatmap(sorted_data, cmap='coolwarm', annot=True, cbar=True, 
                linewidths=0.5, linecolor='black', annot_kws={"size": 10}, 
                xticklabels=False, yticklabels=True)
    plt.title(f'Heatmap of Mutual Information for {feature} (Adjusted)')
    plt.yticks(rotation=0)  # Rotate y-axis labels to be horizontal
    plt.xticks(rotation=90)  # Rotate x-axis labels to be vertical
    plt.xlabel('Features')
    plt.ylabel('Mutual Information')
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.savefig(f'Correlation/2/Heatmap/heatmap2_100_{feature}.png')
    plt.close()