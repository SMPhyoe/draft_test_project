# Created by Su Myat Phyoe at 8:22 am 15/3/2024 using PyCharm
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Predict
y_pred = voting_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Voting Classifier Accuracy:", accuracy)
