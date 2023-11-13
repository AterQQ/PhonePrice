import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from packages.transformers import LowerCaseTransformer

# Load the dataset
file_path = '../data/cleaned_all_phones.csv'
phone_data = pd.read_csv(file_path)

# Dropping non-relevant features
features_to_drop = ['phone_name', 'resolution']
phone_data = phone_data.drop(features_to_drop, axis=1)

# Selecting features and target variable
X = phone_data.drop('price(USD)', axis=1)
y = phone_data['price(USD)']

# Defining numerical and categorical columns
numerical_cols = ['inches', 'ram(GB)', 'weight(g)', 'storage(GB)']
categorical_cols = ['brand', 'video_720p', 'video_1080p', 'video_4K', 'video_8K', 'video_30fps', 'video_60fps', 'video_120fps', 'video_240fps', 'video_480fps', 'video_960fps']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), #
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('lower_case', LowerCaseTransformer()),  # Converts all strings to lower case
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Applying preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Convert to dense format if necessary
X_train_transformed = X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed
X_test_transformed = X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed

# Manual train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_transformed, y_train, test_size=0.2, random_state=1)
print(X_train_split.shape)
print(X_val_split.shape)
print(y_train_split.shape)
print(y_val_split.shape)
# Build the ANN model
def build_model(input_shape):
    model = Sequential([
        Dense(37, activation='tanh', input_shape=(input_shape,)),
        Dense(74, activation='gelu'),
        #Dense(74, activation='relu'),
        #Dropout(0.1),
        Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer= 'adam', loss='mean_squared_error')
    return model

# Create the ANN model
ann_model = build_model(X_train_split.shape[1])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# model summary
ann_model.summary()

# Train the model
history = ann_model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=1000,
    callbacks=[early_stopping],
    batch_size=16
)

# Evaluate the model on the test data
ann_mse = ann_model.evaluate(X_test_transformed, y_test, verbose=0)
print("Mean Squared Error on Test Data:", ann_mse)

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('./ann_loss.png')

# Save the model
model_save_path = "./saved_ann"
ann_model.save(model_save_path)

# Save the preprocessor
preprocessor_save_path = "./saved_preprocessor.pkl"
joblib.dump(preprocessor, preprocessor_save_path)
