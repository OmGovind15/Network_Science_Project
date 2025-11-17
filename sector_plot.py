import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\91878\Documents\projects\network_science_project\sector_analysis.csv'
sector_data = pd.read_csv(file_path)

# Sort the data by 'Avg. Degree' for a clean plot
# We sort ascending=True so that when plotted with barh,
# the highest value is at the top.
sector_data_sorted = sector_data.sort_values('Avg. Degree', ascending=True)

# Create the horizontal bar chart
plt.figure(figsize=(10, 8))
plt.barh(sector_data_sorted['Industry'], sector_data_sorted['Avg. Degree'], color='skyblue')

# Add labels and title
plt.xlabel('Average Degree')
plt.ylabel('Industry')
plt.title('Network Connectivity by Sector (theta=0.5)')

# Ensure layout is tight to prevent label cutoff
plt.tight_layout()

# Save the plot
output_plot_file = 'sector_connectivity_bar_chart.png'
plt.savefig(output_plot_file)