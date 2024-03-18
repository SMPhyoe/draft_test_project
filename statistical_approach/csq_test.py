# Created by Su Myat Phyoe at 8:00 pm 14/3/2024 using PyCharm

import numpy as np
from scipy.stats import chi2_contingency

# Create contingency table (2x2)
observed = np.array([[50, 30], [20, 60]])

'''Gather contingency tables from your dataset at different time intervals.
Apply the chi-squared test to compare pairs of contingency tables collected at different time intervals.
A significant chi-squared statistic and low p-value suggest that the distributions of categorical variables have 
changed over time, indicating potential data drift.'''

# Perform Chi-squared test
chi2_stat, p_value, dof, expected = chi2_contingency(observed)
print("Chi-squared Test:")
print("Chi-squared statistic:", chi2_stat)
print("p-value:", p_value)
