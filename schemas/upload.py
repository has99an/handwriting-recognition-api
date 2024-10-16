from pydantic import BaseModel

class UploadCreate(BaseModel):
    user_id: int
    image: str  # Kan være en URL eller sti til billedet
    pdf: str    # Kan være en URL eller sti til PDF-filen

class UploadResponse(BaseModel):
    id: int
    user_id: int
    image: str
    pdf: str

    class Config:
        orm_mode = True  # Tillad at konvertere ORM-objekter til JSON
