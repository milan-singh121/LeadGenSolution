"""
This file defines and configures the central Celery application instance.
Workers will import this `app` object to connect to the Celery network.
"""

import os
from celery import Celery

# --- Configuration ---
REDIS_BROKER_URL = os.environ.get("REDIS_BROKER_URL", "redis://localhost:6379/0")

# --- Celery App Initialization ---
app = Celery(
    "lead_gen_tasks",
    broker=REDIS_BROKER_URL,
    backend=REDIS_BROKER_URL,
    include=["tasks"],  # Explicitly tell Celery where to find the task definitions.
)

# --- Optional Configuration ---
app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

# app.autodiscover_tasks()

if __name__ == "__main__":
    app.start()
