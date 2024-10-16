from sqlalchemy.orm import Session
from models import Upload
from schemas.upload import UploadCreate

def create_upload(upload: UploadCreate, db: Session):
    new_upload = Upload(
        user_id=upload.user_id,
        image=upload.image,
        pdf=upload.pdf
    )
    db.add(new_upload)
    db.commit()
    db.refresh(new_upload)
    return new_upload

def get_user_uploads(user_id: int, db: Session):
    return db.query(Upload).filter(Upload.user_id == user_id).all()
