# Created by Su Myat Phyoe at 8:13 pm 14/3/2024 using PyCharm
import numpy as np
from scipy.stats import entropy

# Generate two probability distributions
distribution1 = np.array([0.2, 0.3, 0.5])
distribution2 = np.array([0.3, 0.4, 0.3])

'''Collect probability distributions from your dataset at different time intervals.
Calculate the KL divergence between pairs of probability distributions collected at different time intervals.
A high KL divergence value suggests a significant difference between the distributions, indicating potential data 
drift.'''

# Calculate KL divergence
kl_divergence = entropy(distribution1, distribution2)
print("KL Divergence Test:")
print("KL Divergence:", kl_divergence)

