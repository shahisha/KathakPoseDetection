import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

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

# Convert the sparse matrix to a dense NumPy array 
X_train_array = X_train_encoded.toarray()

# Transform the X_test data
X_test_encoded = encoder.transform(X_test)
X_test_array = X_test_encoded.toarray()

# Hyperparameters grid for Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

# Perform grid search for Gradient Boosting
gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=3, scoring='accuracy')
gb_grid_search.fit(X_train_array, y_train)

# Get the best Gradient Boosting model
best_gb_model = gb_grid_search.best_estimator_

# Predict using the best Gradient Boosting model
y_pred_gb = best_gb_model.predict(X_test_array)

# Calculate metrics for Gradient Boosting
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb, average='weighted')
recall_gb = recall_score(y_test, y_pred_gb, average='weighted')
f1_gb = f1_score(y_test, y_pred_gb, average='weighted')

print("Gradient Boosting Metrics:")
print(f"Accuracy: {accuracy_gb:.2f}")
print(f"Precision: {precision_gb:.2f}")
print(f"Recall: {recall_gb:.2f}")
print(f"F1-score: {f1_gb:.2f}")

