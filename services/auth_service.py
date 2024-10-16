from sqlalchemy.orm import Session
from models.user import User
from services.security import get_password_hash, verify_password, create_access_token
from schemas.auth import UserCreate, UserLogin

def register_user(user: UserCreate, db: Session):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        return None  # Bruger findes allerede

    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def login_user(user: UserLogin, db: Session):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        return None  # Ugyldige legitimationsoplysninger

    access_token = create_access_token(data={"sub": db_user.email})
    return access_token
