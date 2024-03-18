# Created by Su Myat Phyoe at 11:31 pm 17/3/2024 using PyCharm
import time

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train initial model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Periodically collect predictions from the model
while True:
    # Collect predictions from the model
    predictions1 = classifier.predict(X_test)

    # Retrain the model with new data (not shown)

    # Collect predictions from the updated model
    predictions2 = classifier.predict(X_test)

    # Compare prediction distributions
    statistic, p_value = ks_2samp(predictions1, predictions2)

    # Check for potential data drift
    if p_value < 0.05:  # Example significance level
        print("Potential data drift detected! Significant difference in prediction distribution.")
        # Trigger alert or take corrective action

    # Sleep for specified interval before next evaluation
    time.sleep(86400)  # Example: 24 hours
