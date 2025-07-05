"""
This script provides a simple, reusable client for interacting with MongoDB.
It is designed to be instantiated within a task, removing the need for a Singleton pattern.
"""

import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    A standard MongoDB client class for connecting to a specified database.

    This class is designed to be instantiated where needed (e.g., at the start
    of a Celery task) with the required connection details. This avoids global
    state and makes the component more modular and testable.
    """

    def __init__(self, uri: str, db_name: str):
        """
        Initializes the MongoDB client and connects to the database.

        Args:
            uri (str): The MongoDB connection URI.
            db_name (str): The name of the database to connect to.

        Raises:
            ConfigurationError: If the URI or database name are invalid.
            ConnectionFailure: If the client cannot connect to the MongoDB server.
        """
        if not uri or not isinstance(uri, str):
            raise ConfigurationError("MongoDB URI must be a valid string.")
        if not db_name or not isinstance(db_name, str):
            raise ConfigurationError("MongoDB database name must be a valid string.")

        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

        try:
            # Establish the connection
            self.client = MongoClient(self.uri, maxPoolSize=50, minPoolSize=10)

            # The ismaster command is cheap and does not require auth. It's a simple
            # way to verify that the server is reachable.
            self.client.admin.command("ismaster")

            self.db = self.client[self.db_name]
            logger.info(f"Successfully connected to MongoDB database: '{self.db_name}'")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB at {self.uri}. Error: {e}")
            raise ConnectionFailure(f"Could not connect to MongoDB: {e}") from e
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during MongoDB initialization: {e}"
            )
            raise

    def get_collection(self, collection_name: str):
        """
        Retrieves a collection from the connected database.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            A PyMongo Collection object.
        """
        if not collection_name:
            raise ValueError("Collection name cannot be empty.")
        return self.db[collection_name]

    def close(self):
        """
        Closes the MongoDB connection if it's open.
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")
