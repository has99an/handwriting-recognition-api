from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from models.user import User
from services.auth_service import verify_password, get_password_hash, create_access_token
from database.db import get_db
from schemas.auth import UserCreate, UserLogin

auth_router = APIRouter()

# Register a new user
@auth_router.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    # Hash password and create a new user
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # Get the generated user ID

    # Create an access token
    access_token = create_access_token(data={"sub": new_user.email})

    # Return the user ID and token
    return {
        "message": "User registered successfully",
        "user_id": new_user.id,
        "access_token": access_token,
        "token_type": "bearer"
    }

# Log in an existing user
@auth_router.post("/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    # Check if the user exists and password is valid
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    # Create an access token
    access_token = create_access_token(data={"sub": db_user.email})

    # Return the user ID and token
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": db_user.id
    }
