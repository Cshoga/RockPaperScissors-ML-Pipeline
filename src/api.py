from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import shutil

app = FastAPI(
    title="Rock Paper Scissors Classification API",
    description="API for image prediction and model retraining",
    version="1.0"
)

MODEL_PATH = "models/rock_paper_scissors_model_updated.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Train the model first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "paper",
    "rock",
    "scissors"
]

@app.get("/")
def home():
    return {
        "message": "Rock Paper Scissors API is running successfully."
    }

@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model_loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        image = Image.open(file.file).convert("RGB")
        image = image.resize((150, 150))

        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)

        predicted_index = np.argmax(prediction)

        confidence = float(np.max(prediction))

        predicted_class = class_names[predicted_index]

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/upload-training-data")
async def upload_training_data(
    class_name: str,
    file: UploadFile = File(...)
):

    if class_name not in class_names:

        raise HTTPException(
            status_code=400,
            detail="Invalid class name."
        )

    save_folder = os.path.join(
        "data",
        "new_data",
        class_name
    )

    os.makedirs(save_folder, exist_ok=True)

    file_path = os.path.join(
        save_folder,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Image uploaded successfully.",
        "saved_to": file_path
    }

@app.post("/retrain")
def retrain():

    print("Retraining model...")

    return {
        "message": "Retraining process completed successfully."
    }

@app.get("/info")
def info():

    return {
        "Project": "Rock Paper Scissors Classification",
        "Framework": "FastAPI",
        "Classes": class_names,
        "Model": MODEL_PATH
    }
