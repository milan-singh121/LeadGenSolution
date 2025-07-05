"""
This file defines and configures the central Celery application instance.
Workers will import this `app` object to connect to the Celery network.
"""

import os
from celery import Celery
import logging
import ssl
import base64
import tempfile

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
REDIS_BROKER_URL = os.environ.get("REDIS_BROKER_URL")
REDIS_CA_CERT_B64 = os.environ.get("REDIS_CA_CERT_B64")

if not REDIS_BROKER_URL:
    raise ValueError("FATAL: REDIS_BROKER_URL environment variable not set.")

if not REDIS_CA_CERT_B64:
    raise ValueError("FATAL: REDIS_CA_CERT_B64 environment variable not set.")

logging.info(f"Using Redis Broker URL: {REDIS_BROKER_URL}")

# Decode the base64 CA cert and write to temp file
with tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=".pem") as cert_file:
    cert_file.write(base64.b64decode(REDIS_CA_CERT_B64))
    ca_cert_path = cert_file.name

logger.info(f"CA certificate written to temporary path: {ca_cert_path}")

# --- Celery App Initialization ---
celery_app = Celery(
    "workspace",
    broker=REDIS_BROKER_URL,
    backend=REDIS_BROKER_URL,
    include=["tasks"],
)

# --- Secure SSL Config ---
celery_app.conf.broker_use_ssl = {
    "ssl_cert_reqs": ssl.CERT_REQUIRED,
    "ssl_ca_certs": ca_cert_path,
}
celery_app.conf.redis_backend_use_ssl = {
    "ssl_cert_reqs": ssl.CERT_REQUIRED,
    "ssl_ca_certs": ca_cert_path,
}

# --- Optional Configuration ---
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

# Automatically discover tasks
celery_app.autodiscover_tasks()

if __name__ == "__main__":
    celery_app.start()
