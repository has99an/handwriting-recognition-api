from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from services.upload_service import process_image, create_pdf
from database.db import get_db
from models.upload import Upload
from fastapi.responses import StreamingResponse, JSONResponse
import io

router = APIRouter()

@router.post("/upload")
async def upload_image(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Process image, generate PDF
    processed_image, pdf_data = process_image(file.file)

    # Save upload record in database
    new_upload = Upload(user_id=user_id, image=processed_image, pdf=pdf_data)
    db.add(new_upload)
    db.commit()

    return StreamingResponse(io.BytesIO(pdf_data), media_type='application/pdf')


@router.get("/history")
def get_upload_history(user_id: int, db: Session = Depends(get_db)):
    uploads = db.query(Upload).filter(Upload.user_id == user_id).all()
    if not uploads:
        return JSONResponse(content={'error': 'No history found.'}, status_code=404)
    return uploads
