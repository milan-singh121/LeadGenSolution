"""
This file defines and configures the central Celery application instance.
Workers will import this `app` object to connect to the Celery network.
"""

import os
from celery import Celery
import logging
import ssl
import base64
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
REDIS_BROKER_URL = os.environ.get("REDIS_BROKER_URL")
REDIS_CA_CERT_B64 = os.environ.get("REDIS_CA_CERT_B64")

CERT_PATH = "redis-ca-cert.pem"

if not REDIS_BROKER_URL:
    raise ValueError("FATAL: REDIS_BROKER_URL environment variable not set.")

broker_ssl_config = None
# Check if the connection URL requires SSL and if the certificate is provided
if REDIS_BROKER_URL.startswith("rediss://") and REDIS_CA_CERT_B64:
    try:
        # Decode the base64 CA cert and write to a file in the project directory
        cert_bytes = base64.b64decode(REDIS_CA_CERT_B64)
        with open(CERT_PATH, "wb") as cert_file:
            cert_file.write(cert_bytes)
        logger.info(f"CA certificate written to path: {CERT_PATH}")

        # --- Secure SSL Config ---
        broker_ssl_config = {
            "ssl_cert_reqs": ssl.CERT_REQUIRED,
            "ssl_ca_certs": CERT_PATH,
        }
    except Exception as e:
        logger.error(f"Failed to process CA certificate, proceeding without SSL: {e}")
        broker_ssl_config = None
else:
    logger.info("Connecting to Redis without SSL (or certificate not provided).")


# --- Celery App Initialization ---
celery_app = Celery(
    "celery_app",  # More conventional to name the app after its own module.
    broker=REDIS_BROKER_URL,
    backend=REDIS_BROKER_URL,
    include=["tasks"],  # CRITICAL: Explicitly include the tasks module.
)

# Apply SSL configuration if it was created successfully
if broker_ssl_config:
    celery_app.conf.broker_use_ssl = broker_ssl_config
    celery_app.conf.redis_backend_use_ssl = broker_ssl_config

# --- Optional Configuration ---
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)
