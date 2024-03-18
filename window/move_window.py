# Created by Su Myat Phyoe at 8:09 am 14/3/2024 using PyCharm

from scipy.stats import ks_2samp
import numpy as np

# Generate data stream (e.g., daily sales data)
data_stream = np.random.normal(loc=0, scale=1, size=1000)

window_size = 100  # Define the size of the moving window
threshold = 0.05  # Define the significance threshold for the KS test

for i in range(window_size, len(data_stream)):
    window1 = data_stream[i - window_size:i]
    window2 = data_stream[i:i + window_size]

    statistic, p_value = ks_2samp(window1, window2)

    if p_value < threshold:
        print(f"Data drift detected at index {i} with KS statistic {statistic} and p-value {p_value}")
