import streamlit as st
import cv2
from huggingface_hub import from_pretrained_keras
from PIL import Image
import numpy as np

ROWS, COLS = 150, 150

model = from_pretrained_keras("carlosaguayo/cats_vs_dogs")


def process_image(img):
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    img = img.reshape(1, ROWS, COLS, 3)

    prediction = model.predict(img)[0][0]
    if prediction >= 0.5:
        message = 'I am {:.2%} sure this is a Cat'.format(prediction)
    else:
        message = 'I am {:.2%} sure this is a Dog'.format(1 - prediction)
    return message


# Webapp

st.set_page_config(
    page_title="Cat vs dog classify",
    page_icon="🐶"
)

st.title('🐶 Simple Cat vs Dog classification')

st.write('Choose a file')

col1, col2 = st.columns([2, 1])

image_placeholder = col1.empty()

uploaded_file = col1.file_uploader('Choose a file', type=['png', 'jpg', 'jpeg'], label_visibility='collapsed')

button_placeholder = col2.empty()
button = button_placeholder.button('Classify', type='primary', disabled=uploaded_file is None)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_placeholder.image(image, use_column_width=True)

    if button:
        with button_placeholder:
            with st.spinner('Processing...'):
                col2.subheader(process_image(np.array(image)))

st.caption('by Vladislav Shalnev')
