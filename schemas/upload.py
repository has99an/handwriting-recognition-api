from pydantic import BaseModel
from datetime import datetime

class UploadCreate(BaseModel):
    user_id: int
    image: str  # Kan være en URL eller sti til billedet
    pdf: str    # Kan være en URL eller sti til PDF-filen

class UploadResponse(BaseModel):
    id: int
    user_id: int
    image: str
    pdf: str
    created_at: datetime  # Tilføjet for at inkludere oprettelsestidspunktet

    class Config:
        orm_mode = True  # Tillad at konvertere ORM-objekter til JSON
