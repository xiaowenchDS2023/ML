import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Load logistic regression model
logistic_regression_model_path = "C:/Users/armen/Downloads/EC-utbildning/2024-V.7-Machine learning/Xiaowen_Chen inlämningsuppgifter/logistic_regression_model.pkl"
logistic_regression_model = joblib.load(logistic_regression_model_path)

# Load random forest model
random_forest_model_path = "C:/Users/armen/Downloads/EC-utbildning/2024-V.7-Machine learning/Xiaowen_Chen inlämningsuppgifter/random_forest_model.pkl"
random_forest_model = joblib.load(random_forest_model_path)

# Load SVM model
svm_model_path = "C:/Users/armen/Downloads/EC-utbildning/2024-V.7-Machine learning/Xiaowen_Chen inlämningsuppgifter/svm_model.pkl"
svm_model = joblib.load(svm_model_path)

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    # Resize the image to 28x28 (MNIST digit size)
    resized_image = grayscale_image.resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(resized_image)
    # Flatten the image array
    flattened_image = image_array.flatten()
    # Normalize the pixel values
    normalized_image = flattened_image / 255.0
    # Return the preprocessed image
    return normalized_image

# Main title
st.title('Digit Recognition App')

# File uploader for image input
uploaded_image = st.file_uploader("Upload a digit image", type=["jpg", "png"])

# When an image is uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded digit', use_column_width=True)

    # Preprocess the uploaded image
    processed_image = preprocess_image(Image.open(uploaded_image))

    # Make prediction using logistic regression model
    lr_prediction = logistic_regression_model.predict([processed_image])[0]

    # Make prediction using random forest model
    rf_prediction = random_forest_model.predict([processed_image])[0]

    # Make prediction using SVM model
    svm_prediction = svm_model.predict([processed_image])[0]

    # Display the predictions
    st.write("Logistic Regression Prediction:", lr_prediction)
    st.write("Random Forest Prediction:", rf_prediction)
    st.write("SVM Prediction:", svm_prediction)
