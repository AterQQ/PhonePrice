
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('../train.csv')
df.head()


# Data Preprocessing

# Since our dataset does not contain any categorical variables that need encoding,
# we will focus on splitting the data and scaling.

# Splitting the dataset into features and target variable
X = df.drop('price_range', axis=1)
y = df['price_range']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
log_reg_pred = log_reg.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))
print(confusion_matrix(y_test, log_reg_pred))
print(classification_report(y_test, log_reg_pred))

# Save the model
joblib.dump(log_reg, './saved_log_reg_model/model.pkl')
print("Model dumped!")

# Save the scaler
joblib.dump(scaler, './saved_log_reg_model/scaler.pkl')
print("Scaler dumped!")