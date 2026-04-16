from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "VocaCare"
    DATABASE_NAME: str = "Stuttering"
    MONGODB_URL: str = Field(..., description="MongoDB connection string (store in environment, not in code)")
    SECRET_KEY: str = Field(..., min_length=32, description="JWT secret (store in environment, not in code)")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
