from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('training/model.keras')

# Create a label map
label_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F',
    16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L',
    22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
    34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
    40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p',
    52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
    58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the handwriting recognition API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded image file
    image = Image.open(file.file)

    # Preprocess the image (resize and convert to grayscale if needed)
    image = image.resize((32, 32))  # Resizing to 32x32 pixels
    image = image.convert('L')  # Convert to grayscale
    input_data = np.array(image).reshape(1, 32, 32, 1)  # Reshape for the model

    # Normalize the image data
    input_data = input_data.astype('float32') / 255.0  # Normalize to [0, 1]

    # Make a prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)

    # Convert predicted class index to letter
    predicted_letter = label_map[int(predicted_class[0])]

    # Return the predicted letter
    return JSONResponse(content={'predicted_letter': predicted_letter})

# Run the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
