"""
This script provides a refactored, reusable client for interacting with the Snov.io API.
It is designed for use in a decoupled, task-based architecture.
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any, List

import requests

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class SnovError(Exception):
    """Custom exception for Snov.io client errors."""

    pass


class SnovClient:
    """
    A client for interacting with the Snov.io REST API.

    This class handles token management and provides methods for the various
    Snov.io endpoints. It is designed to be instantiated within a Celery task.
    """

    BASE_URL = "https://api.snov.io"

    def __init__(self, client_id: str, client_secret: str):
        """
        Initializes the Snov.io client.

        Args:
            client_id (str): Your Snov.io API User ID.
            client_secret (str): Your Snov.io API Secret.
        """
        if not client_id or not client_secret:
            raise ValueError("Snov.io client_id and client_secret cannot be empty.")

        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: Optional[str] = None

    def _get_access_token(self) -> str:
        """
        Generates or retrieves a cached access token for the Snov.io API.
        """

        url = f"{self.BASE_URL}/v1/oauth/access_token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            token_data = response.json()

            if "access_token" not in token_data:
                raise SnovError(
                    f"Failed to get access token from Snov.io. Response: {token_data}"
                )

            self._access_token = token_data["access_token"]

            logger.info("Successfully obtained new Snov.io access token.")
            return self._access_token

        except requests.RequestException as e:
            logger.error(f"Error requesting Snov.io access token: {e}")
            raise SnovError(
                f"Could not connect to Snov.io to get access token: {e}"
            ) from e

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """A helper method to make authenticated requests to the Snov.io API."""
        url = f"{self.BASE_URL}{endpoint}"

        # Snov.io V1 uses access_token in params/data, V2 uses Bearer token
        if "/v2/" in endpoint:
            headers = kwargs.get("headers", {})
            headers["Authorization"] = f"Bearer {self._get_access_token()}"
            kwargs["headers"] = headers
        else:  # V1 logic
            if "params" in kwargs:
                kwargs["params"]["access_token"] = self._get_access_token()
            elif "data" in kwargs:
                kwargs["data"]["access_token"] = self._get_access_token()
            else:  # Default to params
                kwargs["params"] = {"access_token": self._get_access_token()}

        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            error_text = e.response.text if e.response else "No response body"
            logger.error(f"Snov.io HTTP Error for {url}: {e} | Response: {error_text}")
            raise SnovError(
                f"Snov.io API returned status {e.response.status_code}: {error_text}"
            ) from e
        except requests.RequestException as e:
            logger.error(f"Request to Snov.io endpoint {url} failed: {e}")
            raise SnovError(f"Request to Snov.io failed: {e}") from e

    def get_user_lists(self):
        token = self._get_access_token()
        params = {"access_token": token}

        res = requests.get("https://api.snov.io/v1/get-user-lists", params=params)  #

        return json.loads(res.text)

    def create_prospect_list(self, name):
        token = self._get_access_token()
        params = {
            "name": name,
            "access_token": token,
        }

        res = requests.post("https://api.snov.io/v1/lists", data=params)

        return json.loads(res.text)

    def add_prospect_to_list(
        self, list_id: int, prospect_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adds a single prospect to a specified list with their details and custom fields.

        Args:
            list_id (int): The ID of the Snov.io list to add the prospect to.
            prospect_data (Dict[str, Any]): A dictionary containing the prospect's data.
                Example:
                {
                    "email": "test@example.com",
                    "firstName": "John",
                    "lastName": "Doe",
                    "country": "USA",
                    "position": "CEO",
                    "companyName": "Example Corp",
                    "socialLinks[linkedIn]": "linkedin.com/in/johndoe",
                    "customFields[Question 1]": "Answer 1",
                    ...
                }
        """
        payload = prospect_data.copy()
        payload["listId"] = list_id
        payload["updateContact"] = 1  # Update if exists

        return self._make_request("POST", "/v1/add-prospect-to-list", data=payload)

    def get_profile_by_url(self, profile_url: str) -> Dict[str, Any]:
        """Fetches a prospect's profile from Snov.io using their LinkedIn URL."""
        self._make_request("POST", "/v1/add-url-for-search", data={"url": profile_url})
        time.sleep(15)
        return self._make_request(
            "POST", "/v1/get-emails-from-url", data={"url": profile_url}
        )

    def find_emails_by_name_and_domain(
        self, first_name: str, last_name: str, domain: str
    ) -> Dict[str, Any]:
        """
        Starts a task to find an email by name and domain.
        """
        payload = {
            "rows": [
                {"first_name": first_name, "last_name": last_name, "domain": domain}
            ]
        }
        return self._make_request(
            "POST", "/v2/emails-by-domain-by-name/start", json=payload
        )

    def get_email_finder_result(self, task_hash: str) -> Dict[str, Any]:
        """
        Retrieves the result of an email finder task.
        """
        params = {"task_hash": task_hash}
        return self._make_request(
            "GET", "/v2/emails-by-domain-by-name/result", params=params
        )
