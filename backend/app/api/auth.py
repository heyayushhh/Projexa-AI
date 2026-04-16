from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from ..utils.models import UserCreate, UserResponse, UserInDB
from ..utils.db import get_db
import bcrypt
from datetime import datetime, timedelta
import jwt
from ..utils.config import settings

router = APIRouter()

def get_password_hash(password: str):
    # Hash a password for the first time
    # (bcrypt requires bytes, so we encode the string)
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str):
    # Check hashed password. Using .encode('utf-8') to get bytes.
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    db = get_db()
    existing_user = await db.users.find_one({"$or": [{"email": user.email}, {"username": user.username}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    user_dict = user.dict()
    user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
    user_dict["created_at"] = datetime.utcnow()
    
    new_user = await db.users.insert_one(user_dict)
    user_dict["_id"] = str(new_user.inserted_id)
    return user_dict

@router.post("/login")
async def login(form_data: dict): # Simplified for now
    db = get_db()
    user = await db.users.find_one({"$or": [{"email": form_data.get("username")}, {"username": form_data.get("username")}]})
    
    if not user or not verify_password(form_data.get("password"), user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": str(user["_id"]), "username": user["username"]})
    return {"access_token": token, "token_type": "bearer", "user": {
        "id": str(user["_id"]),
        "username": user["username"],
        "name": user["name"],
        "email": user["email"]
    }}
