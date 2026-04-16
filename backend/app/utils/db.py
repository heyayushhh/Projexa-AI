from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings
import certifi
import logging

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    try:
        db.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            tls=True,
            retryWrites=True,
        )
        await db.client.admin.command('ping')
        db.db = db.client[settings.DATABASE_NAME]
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.exception("Failed to connect to MongoDB")
        raise

async def close_mongo_connection():
    if db.client:
        db.client.close()
        logger.info("MongoDB connection closed")

def get_db():
    return db.db
