import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import os

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Rock Paper Scissors Classifier",
    layout="wide"
)

st.title("Rock Paper Scissors Classification")

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Prediction",
        "Model Status",
        "Upload Training Data",
        "Retrain Model",
        "Visualizations"
    ]
)

# ---------------------------------------
# Prediction
# ---------------------------------------

if menu == "Prediction":

    st.header("Predict Hand Gesture")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image,width=300)

        if st.button("Predict"):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                API_URL + "/predict",
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    f"Prediction: {result['prediction']}"
                )

                st.info(
                    f"Confidence: {result['confidence']:.2f}"
                )

            else:
                st.error(f"Prediction failed! Status code: {response.status_code}")

                try:
                    st.write(response.json())
                except Exception:
                    st.write(response.text)

# ---------------------------------------
# Model Status
# ---------------------------------------

elif menu == "Model Status":

    st.header("API Health")

    response = requests.get(
        API_URL + "/health"
    )

    if response.status_code == 200:

        st.success(response.json())

    else:

        st.error("API Offline")

# ---------------------------------------
# Upload Images
# ---------------------------------------

elif menu == "Upload Training Data":

    st.header("Upload Images for Retraining")

    class_name = st.selectbox(
        "Class",
        ["paper","rock","scissors"]
    )

    uploaded_files = st.file_uploader(
        "Upload Images",
        accept_multiple_files=True,
        type=["jpg","png","jpeg"]
    )

    if st.button("Upload Images"):

        for image in uploaded_files:

            requests.post(

                API_URL + "/upload-training-data",

                params={
                    "class_name":class_name
                },

                files={
                    "file":(
                        image.name,
                        image.getvalue()
                    )
                }

            )

        st.success("Images Uploaded Successfully!")

# ---------------------------------------
# Retrain
# ---------------------------------------

elif menu == "Retrain Model":

    st.header("Retrain CNN")

    if st.button("Start Retraining"):

        response = requests.post(
            API_URL + "/retrain"
        )

        st.success(
            response.json()["message"]
        )

# ---------------------------------------
# Visualizations
# ---------------------------------------

elif menu == "Visualizations":

    st.header("Dataset Visualizations")

    image_paths = [
        "images/class_distribution.png",
        "images/training_accuracy.png",
        "images/confusion_matrix.png"
    ]

    for image in image_paths:

        if os.path.exists(image):

            st.image(
                image,
                use_container_width=True
            )

        else:

            st.warning(
                f"{image} not found."
            )
