import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
titanic_data = sns.load_dataset("titanic")

# Step 2: Explore the data
print(titanic_data.head())
print(titanic_data.info())

# Step 3: Data Preprocessing
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)
titanic_data['embarked'].fillna(titanic_data['embarked'].mode()[0], inplace=True)
titanic_data.drop('deck', axis=1, inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['sex', 'embarked', 'class', 'who', 'adult_male', 'alone'], drop_first=True)

# Step 4: Feature Selection
features = ['pclass', 'age', 'fare', 'sibsp', 'parch', 'sex_male', 'embarked_Q', 'embarked_S', 'class_Second', 'class_Third', 'who_man', 'who_woman', 'adult_male_True', 'alone_True']
X = titanic_data[features]
y = titanic_data['survived']

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize Features (Optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Build and Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification)

# Step 9: Fine-Tune the Model (Optional)

# Step 10: Make Predictions
new_passenger_data = np.array([[3, 25, 50, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]])  # Replace with your own data
prediction = model.predict(new_passenger_data)
print(f"Predicted Survival: {prediction[0]}")
