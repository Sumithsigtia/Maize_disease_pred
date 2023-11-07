import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
import pickle

model = pickle.load(open('Streamlit_app/model.sav', 'rb'))
CLASS_NAMES = ['Corn-Blight','Corn-Common_Rust','Corn-Healthy']

st.title("Maize(Corn) Plant Leaf Disease Detection")

st.markdown ("Upload an image of the maize (corn) leaf")

plant_image = st.file_uploader("Choose an image...", type="jpg")
submit= st.button('Predict Disease')



if submit:
	if plant_image is not None:
		file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
		opencv_image = cv2.imdecode(file_bytes, 1)
		st.image (opencv_image, channels="BGR")
		st.write (opencv_image.shape)
		opencv_image = cv2.resize (opencv_image, (256,256))
		opencv_image.shape=(-1,256,256,3)
		Y_pred=model.predict (opencv_image)

		result=CLASS_NAMES [np.argmax (Y_pred)]
		y_pred_onehot=tf.keras.utils.to_categorical(Y_pred)
		result=CLASS_NAMES [np.argmax (y_pred_onehot)]
		

		

		st.title (str("This is Maize leaf with "+ result.split('-') [1]))
		if result is 'Corn-Common_Rust':
			st.title(str("Causes for "+result))
			st.write(str("Common rust in maize is caused by a fungal pathogen 				      called Puccinia sorghi. This disease occurs when 					      favorable environmental conditions and susceptible 				      host plants come together. Additional reasons 					      maybe Presence of Fungal Spores, Lack of crop 					      rotation")) 
			st.title(str("Remedies for "+result))
			st.write(str("Use Fertilizers such as Nitrogen, Phosphorus, 					      Potassium, Organic Fertilizers like Compost, 					      manure "))
			st.write(str(" also Monitor Regularly-Early detection 						      allows for prompt action"))
			st.write(str("Nutrient Management-								      Properly nourished plants are more resilient to 					      diseases."))
			st.write(str("Consult Experts-in case of likelihood of 						      severe infections."))

		if result is 'Corn-Healthy':
			st.title(str("To Maintain "+result))
			st.write(str("To maintain the good health of your crop and 					      increase the yield follow Good Agricultural        				      Practices (GAPs) ")) 
			st.write(str("Fertilization and organic manure: Provide balanced 				      and appropriate fertilization based on soil test 					      results."))
			st.write(str("Irrigation Management: Water the maize crop 					      consistently and avoid both overwatering and    					      underwatering."))
			st.write(str("Crop Rotation: Practice crop rotation to break 					     disease cycles and prevent the buildup of pathogens  				     in the soil."))
			st.write(str("Nutrient Management-								      Properly nourished plants are more resilient to 					      diseases."))
			


				      
