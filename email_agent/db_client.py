"""
This script initializes the asynchronous MongoDB client using configuration
settings for the email generation agent.
"""

import os, sys
import motor.motor_asyncio

app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir))  # already project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import MONGO_URI, MONGO_DB_NAME

# Ensure the MONGO_URI is set via the config module
if not MONGO_URI:
    raise ValueError(
        "MONGO_URI not found. Please set it in your .env file and config.py."
    )

# Create a single, reusable client instance
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]

# You can define your collections here for easy access
people_collection = db["People"]
posts_collection = db["Posts"]
about_company_collection = db["AboutCompany"]
blueprints_collection = db["EmailBlueprints"]
clients_collection = db["Clients"]
generated_emails_collection = db["GeneratedEmails"]
prompt_instructions_collection = db["PromptInstructions"]
jobs_collection = db["Jobs"]


async def get_db():
    """Dependency function to get the database client."""
    return db
