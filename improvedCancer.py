import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a pipeline with standardization and SVM
pipeline = make_pipeline(StandardScaler(), SVC())

# Define the parameter grid for GridSearchCV
param_grid = {
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto'],
}

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(x_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc}')

# Print detailed classification report
print(classification_report(y_test, y_pred))