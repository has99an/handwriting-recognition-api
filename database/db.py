from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Indlæs miljøvariabler fra .env filen
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # Få databasen URL fra miljøvariabler

# Opret en database motor
engine = create_engine(DATABASE_URL)

# Opret en session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Opret en base klasse for at definere modeller
Base = declarative_base()

# Dependency for at få database session
def get_db():
    db = SessionLocal()  # Opret en ny session
    try:
        yield db  # Giv sessionen til den kaldende funktion
    finally:
        db.close()  # Luk sessionen, når den ikke længere er nødvendig
