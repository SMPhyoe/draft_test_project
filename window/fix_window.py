# Created by Su Myat Phyoe at 8:11 am 14/3/2024 using PyCharm
import numpy as np

# Generate data stream (e.g., temperature readings)
data_stream = np.random.normal(loc=20, scale=1, size=1000)

window_size = 100  # Define the size of the fixed window
threshold = 2.0  # Define the threshold for detecting drift based on mean and standard deviation

for i in range(window_size, len(data_stream)):
    window = data_stream[i - window_size:i]
    window_mean = np.mean(window)
    window_std = np.std(window)

    if i >= 2 * window_size:
        prev_window = data_stream[i - 2 * window_size:i - window_size]
        prev_window_mean = np.mean(prev_window)
        prev_window_std = np.std(prev_window)

        if abs(window_mean - prev_window_mean) > threshold or abs(window_std - prev_window_std) > threshold:
            print(
                f"Data drift detected at index {i} with mean change {abs(window_mean - prev_window_mean)} and std change {abs(window_std - prev_window_std)}")

