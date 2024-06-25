import pandas as pd

# Load the correlation results CSV file
correlation_results_path = '/mnt/data/attribute_correlations.csv'
correlation_results = pd.read_csv(correlation_results_path)

# Display the contents of the correlation results
import ace_tools as tools; tools.display_dataframe_to_user(name="Correlation Results", dataframe=correlation_results)
