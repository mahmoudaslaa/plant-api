from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Load model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

class_names = [
    'Tomato__Late_blight', 'Tomato_healthy', 'Grape__healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Soybean___healthy',
    'Squash__Powdery_mildew', 'Potato__healthy',
    'Corn_(maize)__Northern_Leaf_Blight', 'Tomato__Early_blight',
    'Tomato__Septoria_leaf_spot', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Strawberry__Leaf_scorch', 'Peach_healthy', 'Apple__Apple_scab',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Bacterial_spot',
    'Apple__Black_rot', 'Blueberry__healthy',
    'Cherry_(including_sour)__Powdery_mildew', 'Peach__Bacterial_spot',
    'Apple__Cedar_apple_rust', 'Tomato_Target_Spot', 'Pepper,_bell__healthy',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Potato___Late_blight',
    'Tomato__Tomato_mosaic_virus', 'Strawberry_healthy', 'Apple__healthy',
    'Grape__Black_rot', 'Potato_Early_blight', 'Cherry(including_sour)___healthy',
    'Corn_(maize)__Common_rust', 'Grape__Esca(Black_Measles)',
    'Raspberry__healthy', 'Tomato__Leaf_Mold',
    'Tomato__Spider_mites Two-spotted_spider_mite', 'Pepper,_bell__Bacterial_spot',
    'Corn_(maize)___healthy'
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image = preprocess_image(await file.read())
        predictions = model.predict(image)
        index = int(np.argmax(predictions))
        classname = class_names[index]
        confidence = float(predictions[0][index])

        if confidence < 0.94:
            classname = 'undefined'

        return {"classname": classname, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

def preprocess_image(file_bytes):
    try:
        img_array = np.frombuffer(file_bytes, np.uint8)
        x = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if x is None:
            raise ValueError("Invalid image")

        x = cv2.resize(x, (160, 160))
        return np.expand_dims(x, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
