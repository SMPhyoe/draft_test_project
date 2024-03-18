# Created by Su Myat Phyoe at 7:52 am 14/3/2024 using PyCharm


from scipy.stats import ks_2samp
import numpy as np
# can use import statistics

# Generate two samples
sample1 = np.random.normal(loc=0, scale=1, size=1000)
sample2 = np.random.normal(loc=0.5, scale=1, size=1000)

'''Collect samples from your dataset at different time intervals (e.g., daily, weekly).
Calculate the Kolmogorov-Smirnov statistic between pairs of samples collected at different time intervals.
If the Kolmogorov-Smirnov statistic is consistently high and low p-value suggest that the distributions of 
categorical variables have changed over time, indicating potential data drift. the p-value is below a chosen 
significance level (e.g., 0.05), it may indicate data drift.'''

# Perform Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(sample1, sample2)
print("Kolmogorov-Smirnov Test:")
print("Statistic:", statistic)
print("p-value:", p_value)
