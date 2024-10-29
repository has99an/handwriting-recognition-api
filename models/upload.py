from sqlalchemy import Column, Integer, ForeignKey, DateTime, TEXT  
from sqlalchemy.orm import relationship
from database.db import Base
from datetime import datetime, timezone, timedelta

class Upload(Base):
    __tablename__ = 'uploads'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    image = Column(TEXT, nullable=False)  # Store Base64 encoded string
    pdf = Column(TEXT, nullable=False)    # Store Base64 encoded string


    created_at = Column(DateTime, default=lambda: datetime.now(timezone(timedelta(hours=2))))  # Automatisk tidsstempel
    user = relationship("User", back_populates="uploads")
