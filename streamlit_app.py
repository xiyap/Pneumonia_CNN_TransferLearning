import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from utils import set_background, classify

set_background('./streamlit_data/background.jpg')

# set title
st.title('Pneumonia Predictor')

# set header
st.write('#### Please upload your chest X-ray image and kindly wait a few seconds for Dr. Ian to review your results.')

# upload file
file = st.file_uploader('', type = ['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./streamlit_data/custom_pneumonia_detection_model_0.8_thres.h5')

# load class names
with open('./streamlit_data/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[-1] for line in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.write('#### Image uploaded:')
    st.image(image, use_column_width = True)

    # classify image
    class_name, y_pred = classify(image, model, class_names)

    # write classification
    st.write('## Result: {}'.format(class_name))
    st.write('#### Based on your submitted X-ray, the percentage of you having pneumonia is {:.2f}%'.format(float((y_pred * 100))))

st.markdown('<style>footer {position: fixed;bottom: 0;width: 100%;}</style>', unsafe_allow_html = True)
st.markdown('###### *Disclaimer: This is not a proper medical diagnosis for Pneumonia.*')
st.markdown('###### *The model has 98.21% accuracy in predicting actual pneumonia diagnosis with 5.90% of pneumonia misdiagnosis.*')