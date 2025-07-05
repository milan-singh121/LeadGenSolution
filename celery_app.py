"""
This file defines and configures the central Celery application instance.
Workers will import this `app` object to connect to the Celery network.
"""

import os
from celery import Celery

# --- Configuration ---
# It's crucial that this is read from the environment variable provided by the App Platform.
REDIS_BROKER_URL = os.environ.get("REDIS_BROKER_URL")

if not REDIS_BROKER_URL:
    # This will cause a clear failure if the environment variable is not set.
    raise ValueError("FATAL: REDIS_BROKER_URL environment variable not set.")

# --- Celery App Initialization ---
# The application name 'LeadGen_Solution' should match your root project folder name.
# ✅ FIX: The `include` argument is removed. We will use autodiscover_tasks instead.
celery_app = Celery(
    "workspace",
    broker=REDIS_BROKER_URL,
    backend=REDIS_BROKER_URL,
)

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
