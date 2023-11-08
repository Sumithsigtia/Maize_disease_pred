import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as keras_image
from PIL import Image

# Load the trained model
model_path = "plant_disease_model_new.h5"  # Replace this with the actual path to your trained model
model = load_model(model_path)

# Define class labels
class_labels = ['Blight', 'Common Rust', 'Healthy']

st.title('Maize Leaf Disease Prediction')

# Function to make predictions on uploaded image
def predict_disease(image):
    img = keras_image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalizing the image

    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    return predicted_class, confidence

# Streamlit app
st.write("Upload a maize leaf image for disease prediction")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Make prediction when the 'Predict' button is clicked
    if st.button('Predict'):
        predicted_class, confidence = predict_disease(img)
        st.write(f'Prediction: {predicted_class}')
        st.write(f'Confidence: {confidence}%')
