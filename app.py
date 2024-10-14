import sys
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Tilføj stien til training mappen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'training')))

from prepare_data import process_image  # Importer process_image funktionen

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

# Global variabel til at gemme det behandlede billede og PDF
processed_image_bytes = None
generated_pdf = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the handwriting recognition API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global processed_image_bytes  # Tilgå den globale variabel

    # Åbn det uploadede billede
    image = Image.open(file.file)

    # Forbehandle billedet
    input_data = process_image(image)

    # Lav en forudsigelse
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)

    # Konverter den forudsagte klasse til bogstav
    predicted_letter = label_map[int(predicted_class[0])]

    # Behandl billedet til returnering
    processed_image = process_image(image)
    processed_image_pil = Image.fromarray((processed_image.squeeze() * 255).astype(np.uint8))

    # Gem det behandlede billede i en BytesIO
    img_byte_arr = io.BytesIO()
    processed_image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    processed_image_bytes = img_byte_arr.getvalue()  # Gem det i den globale variabel

    # Generer PDF
    pdf_data = create_pdf(predicted_letter)
    global generated_pdf
    generated_pdf = pdf_data  # Gem den genererede PDF i en global variabel

    # Returner PDF'en direkte
    return StreamingResponse(io.BytesIO(generated_pdf), media_type='application/pdf', headers={"Content-Disposition": "attachment; filename=recognized_text.pdf"})

def create_pdf(text: str) -> bytes:
    pdf_bytes = io.BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 100, text)
    c.save()
    pdf_bytes.seek(0)
    return pdf_bytes.getvalue()

@app.get("/predict/image")
async def get_processed_image():
    if processed_image_bytes is None:
        return JSONResponse(content={'error': 'No processed image available.'}, status_code=404)
    
    return StreamingResponse(io.BytesIO(processed_image_bytes), media_type='image/png')

# Kør app'en med uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
