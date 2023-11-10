import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as keras_image
from PIL import Image

# Load the trained model
model_path = "plant_disease_model_new.h5"  # Replace this with the actual path to your trained model
model = load_model(model_path)

# Define class labels
class_labels = ['Corn-Blight', 'Corn-Common_Rust', 'Corn-Healthy']

st.title('Maize Leaf Disease Prediction')

# Function to make predictions on uploaded image
def predict_disease(image):
    img = image.resize((256, 256))  # Resize the image to match model input size
    img = keras_image.img_to_array(img)
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

    # Display a smaller version of the image
    displayed_image_size = (300, 300)  # Set the desired size
    st.image(img, caption='Uploaded Image', use_column_width=True, width=displayed_image_size)

    # Make prediction when the 'Predict' button is clicked
    if st.button('Predict'):
        predicted_class, confidence = predict_disease(img)
        st.write(f'Prediction: {predicted_class}')
        st.write(f'Confidence: {confidence}%')

        if predicted_class == 'Corn-Blight':
            st.title("Causes for " + predicted_class)
            st.write("Corn blight in maize can be caused by fungal pathogens such as Helminthosporium maydis or Exserohilum turcicum. These pathogens thrive in warm, humid conditions and can infect corn plants through wounds or natural openings.")
            st.write("Factors such as prolonged leaf wetness, high humidity, and warm temperatures favor the development and spread of corn blight.")

            st.title("Remedies for " + predicted_class)
            st.write("Utilize Resistant Varieties - Planting corn varieties that show resistance to the specific blight-causing pathogens can help mitigate the disease's impact.")
            st.write("Crop Rotation - Rotate with non-host crops to disrupt the disease cycle and reduce the population of blight-causing pathogens.")
            st.write("Fungicides - Application of fungicides, particularly before the onset of favorable conditions for blight, can be an effective management strategy.")
            st.write("Sanitation - Clear fields of crop debris post-harvest to reduce overwintering sites for blight pathogens.")
            st.write("Adequate Spacing - Avoid overcrowding to ensure good air circulation among plants, reducing humidity and disease spread.")
            st.write("Regular Monitoring - Regularly inspect crops for early signs of blight to facilitate prompt action, including removal of infected plants.")
        
        elif predicted_class == 'Corn-Common_Rust':
            st.title("Causes for " + predicted_class)
            st.write("Common rust in maize is caused by a fungal pathogen called Puccinia sorghi. This disease occurs when favorable environmental conditions and susceptible host plants come together. Additional reasons may be the presence of Fungal Spores and lack of crop rotation.")
        
            st.title("Remedies for " + predicted_class)
            st.write("Use Fertilizers such as Nitrogen, Phosphorus, Potassium, and Organic Fertilizers like Compost and manure.")
            st.write("Monitor Regularly - Early detection allows for prompt action.")
            st.write("Nutrient Management - Properly nourished plants are more resilient to diseases.")
            st.write("Consult Experts - In case of the likelihood of severe infections.")
        
        elif predicted_class == 'Corn-Healthy':
            st.title("To Maintain " + predicted_class)
            st.write("To maintain the good health of your crop and increase the yield, follow Good Agricultural Practices (GAPs).")
            st.write("Fertilization and organic manure - Provide balanced and appropriate fertilization based on soil test results.")
            st.write("Irrigation Management - Water the maize crop consistently and avoid both overwatering and underwatering.")
            st.write("Crop Rotation - Practice crop rotation to break disease cycles and prevent the buildup of pathogens in the soil.")
            st.write("Nutrient Management - Properly nourished plants are more resilient to diseases.")
