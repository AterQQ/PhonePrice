import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np
# Load the dataset
file_path = '../data/cleaned_all_phones.csv'
phone_data = pd.read_csv(file_path)

# Define custom price categories
bins = [0, 250, 500, 1000, phone_data['price(USD)'].max()]
labels = ['0-250', '251-500', '501-1000', '1000+']
phone_data['price_category'] = pd.cut(phone_data['price(USD)'], bins=bins, labels=labels, right=False)

# Selecting features and target variable
features = ['storage(GB)', 'video_4K', 'video_720p', 'ram(GB)', 'video_60fps', 'weight(g)', 'inches', 'video_240fps']
X = phone_data[features]
y = phone_data['price_category']

# Preprocessing for numerical data
numerical_cols = X.columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
    ])

# Convert categorical labels to one-hot encoding
y = pd.get_dummies(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Applying preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Convert sparse matrix to dense (if necessary)
X_train_transformed = X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed
X_test_transformed = X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed

# Manual train-validation split
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_transformed, y_train, test_size=0.2, random_state=42)

# Building the ANN for Classification
model = Sequential([
    Dense(8, activation='tanh', input_shape=(X_train_final.shape[1],)),
    Dense(16, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Output layer for the number of categories
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_final, y_train_final,
    epochs=100,
    validation_data=(X_val, y_val)
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_transformed, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plotting training and validation accuracy and loss
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Training acc')
plt.plot(epochs, history.history['val_accuracy'], label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'],  label='Training loss')
plt.plot(epochs, history.history['val_loss'],  label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predicting the Test set results
y_pred = model.predict(X_test_transformed)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test.to_numpy(), axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# F1 Score
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f"F1 Score: {f1}")

# Classification Report (includes F1 score for each class)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))
