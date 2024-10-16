from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database.db import Base


class Upload(Base):
    __tablename__ = 'uploads'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    image = Column(String, nullable=False)  # Sti til det uploadede billede
    pdf = Column(String, nullable=False)     # Sti til den genererede PDF

    user = relationship("User", back_populates="uploads")
