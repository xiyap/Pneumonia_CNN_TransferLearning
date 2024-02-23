import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html = True)


def classify(image, model, class_names):
    resized_image = ImageOps.fit(image, (230, 230), Image.Resampling.LANCZOS)
    resized_image_array = np.asarray(resized_image)
    normalized_image_array = resized_image_array / 255.0
    normalized_image_array = np.expand_dims(normalized_image_array, axis = 0)
    y_pred = model.predict(normalized_image_array)
    y_pred = y_pred.reshape(1, -1)[0]
    threshold = 0.8
    predicted_class_index = int((y_pred > threshold).astype(int))
    class_name = class_names[predicted_class_index]

    return class_name, y_pred