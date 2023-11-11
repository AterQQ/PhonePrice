import numpy as np
import joblib
from keras.models import load_model
import pandas as pd

# Load the saved preprocessor and ANN model
preprocessor = joblib.load("./saved_preprocessor.pkl")  # Adjust the path if needed
model = load_model("./saved_ann")  # Adjust the path if needed

def get_user_input():
    # Gather user input
    screen_size = float(input("Enter Screen Size (in inches): "))
    ram = int(input("Enter RAM (in GB): "))
    weight = float(input("Enter Weight (in grams): "))
    storage = int(input("Enter Storage (in GB): "))
    brand = input("Enter Brand: ").lower()  # Transform brand to lowercase
    video_720p = input("Does it support 720p video? (yes/no): ").lower() == 'yes'
    video_1080p = input("Does it support 1080p video? (yes/no): ").lower() == 'yes'
    video_4k = input("Does it support 4K video? (yes/no): ").lower() == 'yes'
    video_8k = input("Does it support 8K video? (yes/no): ").lower() == 'yes'
    video_30fps = input("Does it support 30fps video? (yes/no): ").lower() == 'yes'
    video_60fps = input("Does it support 60fps video? (yes/no): ").lower() == 'yes'
    video_120fps = input("Does it support 120fps video? (yes/no): ").lower() == 'yes'
    video_240fps = input("Does it support 240fps video? (yes/no): ").lower() == 'yes'
    video_480fps = input("Does it support 480fps video? (yes/no): ").lower() == 'yes'
    video_960fps = input("Does it support 960fps video? (yes/no): ").lower() == 'yes'

    # Create a DataFrame to match the training data format
    return {
        'inches': [screen_size],
        'ram(GB)': [ram],
        'weight(g)': [weight],
        'storage(GB)': [storage],
        'brand': [brand],
        'video_720p': [video_720p],
        'video_1080p': [video_1080p],
        'video_4K': [video_4k],
        'video_8K': [video_8k],
        'video_30fps': [video_30fps],
        'video_60fps': [video_60fps],
        'video_120fps': [video_120fps],
        'video_240fps': [video_240fps],
        'video_480fps': [video_480fps],
        'video_960fps': [video_960fps]
    }

def get_default_input():
    # Returns a predefined set of input values
    return {
        'inches': [6.7],
        'ram(GB)': [6],
        'weight(g)': [254.0],
        'storage(GB)': [512],
        'brand': ['apple'],
        'video_720p': [True],
        'video_1080p': [True],
        'video_4K': [True],
        'video_8K': [False],
        'video_30fps': [True],
        'video_60fps': [True],
        'video_120fps': [True],
        'video_240fps': [True],
        'video_480fps': [False],
        'video_960fps': [False]
    }

def predict_price(input_data):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame.from_dict(input_data)

    # Process the input data
    processed_input = preprocessor.transform(input_df)
    processed_input = processed_input.toarray() if hasattr(processed_input, "toarray") else processed_input

    # Predict the price
    prediction = model.predict(processed_input)
    return prediction[0][0]

def main():
    choice = input("Do you want to use default data? (yes/no): ").lower()
    if choice == 'yes':
        user_input = get_default_input()
    else:
        user_input = get_user_input()

    predicted_price = predict_price(user_input)
    print(f"Predicted Price of the Phone: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
