from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/3")
# try:
#     with open(MODEL, "r") as p:
#         print(json.load(p))
# except json.decoder.JSONDecodeError:
#     print("Error in MODEL file")
CLASS_NAMES = ["Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  # Reads bytes as a PILO image
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  # UploadFile - datatype, File(...) - default value
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[1])]
    confidence = np.max(predictions[1])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
