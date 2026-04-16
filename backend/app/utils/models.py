from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")

class UserBase(BaseModel):
    name: str
    username: str
    email: EmailStr
    description: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserResponse(UserBase):
    id: str = Field(alias="_id")
    created_at: datetime

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}

class SpeechAnalysisBase(BaseModel):
    transcript: str
    fluency_score: float
    total_words: int
    stutterEvents: List[dict] = Field(default_factory=list)
    headMovements: List[dict] = Field(default_factory=list)

class SpeechAnalysisInDB(SpeechAnalysisBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class TrainingSessionBase(BaseModel):
    exercise_id: int
    completed: bool = True
    spoken_text: Optional[str] = None
    expected_text: Optional[str] = None
    fluency_score: Optional[float] = None
    is_correct: Optional[bool] = None
    word_count: Optional[int] = None
    error_count: Optional[int] = None

class TrainingSessionInDB(TrainingSessionBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class TrendDay(BaseModel):
    day: str
    score: float

class DashboardStatsResponse(BaseModel):
    fluencyScore: float
    currentStreak: int
    totalSessions: int
    fluencyTrend: List[TrendDay]
