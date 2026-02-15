# prediction.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('dataset.csv')  # Placeholder for the actual dataset

# Preparing the data
X = data.drop('target', axis=1)  # Features
Y = data['target']  # Target variable

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Making predictions
predictions = model.predict(X_test)

# Function to make a single prediction

def make_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    return model.predict(input_array)  

# Example usage
# single_prediction = make_prediction([value1, value2, ...])  # Replace with actual values