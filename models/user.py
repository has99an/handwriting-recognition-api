# models/user.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from database.db import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(200))

    uploads = relationship("Upload", back_populates="user", lazy='dynamic')  # String reference to Upload
