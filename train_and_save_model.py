import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('expanded_heart_disease_dataset.csv')

# Prepare the dataset for training
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as 'heart_disease_model.pkl'.")
