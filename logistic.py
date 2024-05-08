import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load the CSV file skipping the first row
df = pd.read_csv(r'C:\Users\ishas\Downloads\sign-language-detector-python-master\angles.csv')
# Split the data into features (X) and target (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a one-hot encoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit and transform the X_train data
X_train_encoded = encoder.fit_transform(X_train)

# Use the best parameters obtained from a hypothetical search
best_params = {'max_iter': 1000, 'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}

# Fit the Logistic Regression model with the best parameters
logistic_model = LogisticRegression(**best_params, random_state=42)
logistic_model.fit(X_train_encoded, y_train)

# Transform the X_test data and predict
X_test_encoded = encoder.transform(X_test)
y_pred = logistic_model.predict(X_test_encoded)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

import numpy as np
import matplotlib.pyplot as plt

# Define a range of regularization strengths to evaluate
regularization_strengths = np.logspace(-3, 3, 7)

# Initialize lists to store accuracy scores
accuracy_scores = []

# Iterate over regularization strengths
for C in regularization_strengths:
    # Initialize and fit the Logistic Regression model with the current regularization strength
    logistic_model = LogisticRegression(max_iter=1000, C=C, penalty='l2', solver='liblinear', random_state=42)
    logistic_model.fit(X_train_encoded, y_train)
    
    # Predict using the trained classifier
    y_pred = logistic_model.predict(X_test_encoded)
    
    # Calculate accuracy and append to the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plotting the accuracy scores
plt.plot(regularization_strengths, accuracy_scores, marker='o')
plt.xscale('log')
plt.title('Accuracy vs. Regularization Strength in Logistic Regression')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
