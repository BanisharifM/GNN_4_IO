import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the mutual information values csv file
file_path = 'Correlation/mutual_information2_100.csv'
mutual_info_data = pd.read_csv(file_path, index_col=0)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(mutual_info_data.values.flatten(), bins=50, color='blue', edgecolor='black')
plt.title('Histogram of Mutual Information Values')
plt.xlabel('Mutual Information Value')
plt.ylabel('Frequency')
plt.savefig('Correlation//mutual_information_histogram2_100.png')
plt.close()

# Plotting the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=mutual_info_data.values.flatten(), color='blue')
plt.title('Box Plot of Mutual Information Values')
plt.xlabel('Mutual Information Value')
plt.savefig('Correlation//mutual_information_boxplot2_100.png')
plt.close()

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(mutual_info_data, cmap='viridis')
plt.title('Heatmap of Mutual Information Matrix')
plt.savefig('Correlation//mutual_information_heatmap2_100.png')
plt.close()
