import sys
import os
from fastapi import FastAPI, File, Form, UploadFile, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from sqlalchemy.orm import Session
from models.user import User  # Importer User direkte
from models.upload import Upload  # Importer Upload modellen
from database.db import engine, get_db  # Importer get_db
from routers.auth import auth_router  # Importer auth-routeren
from fastapi.middleware.cors import CORSMiddleware
import base64
from services.upload_service import create_upload
from schemas.upload import UploadCreate


# Tilføj stien til training mappen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'training')))
from prepare_data import process_image  # Importer process_image funktionen

app = FastAPI()
app.include_router(auth_router, prefix="/auth")

# Database initialization
# Sørg for at 'User' modellen er importeret før dette
from models import Base  # Importer Base for at kunne kalde create_all
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Tillad din Angular-app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def predict(user_id: int = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Open the uploaded image
    image = Image.open(file.file)

    # Convert the original image to Base64
    original_image_bytes = io.BytesIO()
    image.save(original_image_bytes, format='PNG')
    original_image_bytes.seek(0)
    original_image_base64 = base64.b64encode(original_image_bytes.getvalue()).decode('utf-8')

    # Process the image
    input_data = process_image(image)

    # Make a prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
  
    # Convert the predicted class to a letter
    predicted_letter = label_map[int(predicted_class[0])]

    # Generate PDF
    pdf_data = create_pdf(predicted_letter)
    
    # Convert PDF to Base64
    generated_pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

    # Save upload in the database with original image
    new_upload = UploadCreate(user_id=user_id, image=original_image_base64, pdf=generated_pdf_base64)
    create_upload(new_upload, db)  # Flytter gemningslogikken her

    # Return the PDF directly
    return StreamingResponse(io.BytesIO(pdf_data), media_type='application/pdf', headers={"Content-Disposition": "attachment; filename=recognized_text.pdf"})


def create_pdf(text: str) -> bytes:
    pdf_bytes = io.BytesIO()
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
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




@app.get("/uploads/{user_id}")
def get_user_uploads(user_id: int, db: Session = Depends(get_db)):
    # Query the uploads based on the user_id
    uploads = db.query(Upload).filter(Upload.user_id == user_id).all()
    
    if not uploads:
        return JSONResponse(content={'error': 'No uploads found for this user.'}, status_code=404)
    
    # Prepare a list of uploads to return
    upload_list = [
        {
            "id": upload.id,
            "user_id": upload.user_id,
            "image": upload.image,  # Optionally, handle how you want to return image info
            "pdf": upload.pdf,      # Optionally, handle how you want to return pdf info
            "created_at": upload.created_at
        } for upload in uploads
    ]
    
    return upload_list

# Kør app'en med uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)