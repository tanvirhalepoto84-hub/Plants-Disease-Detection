# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

model = tf.keras.models.load_model("trained_plant_disease_model.keras")

st.title("🌿 Plant Disease Detector")

uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"🔍 **Prediction:** Class #{predicted_class}")


