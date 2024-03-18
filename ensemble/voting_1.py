# Created by Su Myat Phyoe at 8:18 am 15/3/2024 using PyCharm
import time

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
classifier1 = LogisticRegression()
classifier2 = DecisionTreeClassifier()
classifier3 = SVC()

# Create voting classifier
voting_classifier = VotingClassifier(estimators=[('lr', classifier1), ('dt', classifier2), ('svc', classifier3)])

# Train voting classifier
voting_classifier.fit(X_train, y_train)

# Periodically evaluate model performance and check for data drift
while True:
    # Evaluate model performance
    accuracy = evaluate_model(voting_classifier, X_test, y_test)

    # Check for potential data drift
    if accuracy < 0.8:  # Example threshold
        print("Potential data drift detected! Performance dropped below threshold.")
        # Trigger alert or take corrective action

    # Sleep for specified interval before next evaluation
    time.sleep(86400)  # Example: 24 hours

