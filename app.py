import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
MODEL_PATH = "stroke_model.h5"  # Path to your model
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ["Healthy", "Stroke"]  # Update if your classes differ

# Function to preprocess and predict
def predict_image(uploaded_image, model, class_labels, target_size=(224, 224)):
    img = uploaded_image.resize(target_size)  # Resize image
    img_array = np.array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Scale pixel values

    prediction = model.predict(img_array)
    if len(class_labels) == 2:  # Binary classification
        predicted_label = class_labels[1] if prediction[0][0] > 0.5 else class_labels[0]
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    else:  # Multi-class classification
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        confidence = prediction[0][predicted_class]

    return predicted_label, confidence

# Streamlit app
st.title("Acute Ischemic Stroke Prediction")
st.write("Upload a CT scan image to predict if it is Healthy or indicative of Stroke.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Make prediction
    label, confidence = predict_image(img, model, class_labels)

    # Display results
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}**")
