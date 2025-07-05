"""
This file defines and configures the central Celery application instance.
Workers will import this `app` object to connect to the Celery network.
"""

import os
from celery import Celery
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Configuration ---
# It's crucial that this is read from the environment variable provided by the App Platform.
REDIS_BROKER_URL = os.environ.get("REDIS_BROKER_URL")

if not REDIS_BROKER_URL:
    # This will cause a clear failure if the environment variable is not set.
    raise ValueError("FATAL: REDIS_BROKER_URL environment variable not set.")

logging.info(f"Using Redis Broker URL: {REDIS_BROKER_URL}")

# --- Celery App Initialization ---
# The application name 'LeadGen_Solution' should match your root project folder name.
# This helps Celery with auto-discovery.
celery_app = Celery("workspace", broker=REDIS_BROKER_URL, backend=REDIS_BROKER_URL)

# --- Optional Configuration ---
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

# ✅ FIX: This line tells Celery to automatically find any 'tasks.py' files
# within the project structure. This is the standard way to avoid circular imports.
celery_app.autodiscover_tasks()

if __name__ == "__main__":
    celery_app.start()
