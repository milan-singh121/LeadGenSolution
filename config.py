"""
This file is responsible for loading all configuration and secrets
for the application from environment variables.

It uses the python-dotenv library to load a .env file during local development.
In production (on the droplet), the environment variables will be provided
by the systemd service file.
"""

import os
from dotenv import load_dotenv

# This line loads the environment variables from a .env file.
# It's smart enough to not fail if the file doesn't exist.
load_dotenv()

# --- Secrets and Connection URIs ---
# Load secrets from environment variables. The second argument to os.getenv()
# is a default value, which is useful for preventing errors if a variable is not set.

RAPID_API_KEY = os.getenv("RAPID_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "LeadGenDB")

# Snov.io API Credentials
SNOV_CLIENT_ID = os.getenv("SNOV_CLIENT_ID")
SNOV_CLIENT_SECRET = os.getenv("SNOV_CLIENT_SECRET")

# OpenAI API Credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Claude API Credentials
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

# --- Application-Specific Static Configuration ---

# Example: List of industries to filter out from company search
BLOCKED_INDUSTRIES = ["Staffing and Recruiting", "Human Resources Services"]

# Example: List of titles to search for when looking for prospects
PROSPECT_SEARCH_KEYWORDS = [
    "HR",
    "Human Resource",
    "Talent Acquisition",
    "IT Recruiter",
    "CTO",
    "Chief Technology Officer",
    "Founder",
    "Co-Founder",
    "CEO",
    "Director",
]


# --- Validation (Optional but Recommended) ---
# You can add checks to ensure that critical secrets have been loaded.
def validate_config():
    """Checks that all essential configuration variables are present."""
    critical_vars = {
        "RAPID_API_KEY": RAPID_API_KEY,
        "MONGO_URI": MONGO_URI,
        "SNOV_CLIENT_ID": SNOV_CLIENT_ID,
        "SNOV_CLIENT_SECRET": SNOV_CLIENT_SECRET,
    }
    missing_vars = [key for key, value in critical_vars.items() if not value]
    if missing_vars:
        raise ValueError(
            f"Configuration Error: Missing required environment variables: {', '.join(missing_vars)}"
        )

    print("Configuration loaded and validated successfully.")


# You could call validate_config() at the start of your main application
# or task files to ensure everything is set up before running.
