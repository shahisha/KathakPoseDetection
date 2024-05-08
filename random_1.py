import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle

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

# Fit the Random Forest Classifier with the best parameters
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=8, min_samples_leaf=1, random_state=42)
rf_classifier.fit(X_train_array, y_train)

# Transform the X_test data and predict
X_test_encoded = encoder.transform(X_test)
X_test_array = X_test_encoded.toarray()
y_pred = rf_classifier.predict(X_test_array)

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

