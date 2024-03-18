# Created by Su Myat Phyoe at 8:12 am 15/3/2024 using PyCharm

from scipy.stats import entropy
import numpy as np

# Generate data stream (e.g., stock prices)
data_stream = np.random.normal(loc=100, scale=5, size=1000)

min_window_size = 50  # Define the minimum size of the variable window
max_window_size = 200  # Define the maximum size of the variable window
threshold = 0.1  # Define the threshold for KL divergence

for window_size in range(min_window_size, max_window_size + 1):
    for i in range(window_size, len(data_stream)):
        window = data_stream[i - window_size:i]
        prev_window = data_stream[i - 2 * window_size:i - window_size]

        # Compute histograms with a fixed number of bins
        hist_window, _ = np.histogram(window, bins=10)
        hist_prev_window, _ = np.histogram(prev_window, bins=10)

        # Normalize histograms
        hist_window = hist_window / np.sum(hist_window)
        hist_prev_window = hist_prev_window / np.sum(hist_prev_window)

        # Compute KL divergence
        kl_divergence = entropy(hist_window, hist_prev_window)

        if kl_divergence > threshold:
            print(f"Data drift detected at index {i} with KL divergence {kl_divergence} and window size {window_size}")