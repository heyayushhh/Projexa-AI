from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings
import certifi
import asyncio

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    # Adding certifi for SSL certificates on cloud providers
    # and adding a timeout for better error handling
    try:
        # Use a more stable connection string format for some environments
        # and ensure we use the modern 'tls' parameter instead of 'ssl'
        db.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
            tls=True,
            retryWrites=True
        )
        # Verify connection
        await db.client.admin.command('ping')
        db.db = db.client[settings.DATABASE_NAME]
        print(f"Successfully connected to MongoDB at {settings.MONGODB_URL}")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        # If it fails, we still set db.db to allow the app to start
        # but the lifespan indexes will likely fail too
        db.db = db.client[settings.DATABASE_NAME]
        raise e

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("MongoDB connection closed")

def get_db():
    return db.db
