# Created by Su Myat Phyoe at 11:28 pm 17/3/2024 using PyCharm
import time

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train initial model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Periodically evaluate model performance
while True:
    print('hi')
    # Evaluate model performance
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Check for potential data drift
    if accuracy < 0.8:  # Example threshold
        print("Potential data drift detected! Accuracy dropped below threshold.")
        # Trigger alert or take corrective action

    # Sleep for specified interval before next evaluation
    time.sleep(86400)  # Example: 24 hours
