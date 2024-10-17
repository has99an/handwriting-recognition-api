from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database.db import Base
from datetime import datetime, timezone, timedelta

class Upload(Base):
    __tablename__ = 'uploads'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    image = Column(String(255), nullable=False)
    pdf = Column(String(255), nullable=False)

    # Angiv lokal tidszone (for eksempel CET)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone(timedelta(hours=2))))  # Automatisk udfyldt ved insert

    user = relationship("User", back_populates="uploads")
