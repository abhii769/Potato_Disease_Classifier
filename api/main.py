from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()

# MODEL = tf.keras.models.load_model("../models/1")
MODEL = tf.keras.models.load_model(r"D:\Nothing\Machine Learning\Deep Learning\potato-disease\models\1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/")
async def abc():
    return {"message": "yahoo"}

@app.post("/predict")
async def predicts(
    file: UploadFile=File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = MODEL.predict(img_batch)

    prediced_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions)

    return {
        'class': prediced_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8050)