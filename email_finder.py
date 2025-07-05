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
        self._token_expires_at: float = 0

    def _get_access_token(self) -> str:
        """
        Generates or retrieves a cached access token for the Snov.io API.
        The token is cached to avoid requesting a new one for every API call.

        Returns:
            str: A valid Snov.io access token.

        Raises:
            SnovError: If token generation fails.
        """
        # Check if the token is still valid (with a 60-second buffer)
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        url = f"{self.BASE_URL}/v1/oauth/access_token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        try:
            response = requests.post(url, data=data, timeout=15)
            response.raise_for_status()
            token_data = response.json()

            if "access_token" not in token_data:
                raise SnovError(
                    f"Failed to get access token from Snov.io. Response: {token_data}"
                )

            self._access_token = token_data["access_token"]
            # Snov.io tokens expire in 1 hour (3600 seconds)
            self._token_expires_at = time.time() + token_data.get("expires_in", 3600)

            logger.info("Successfully obtained new Snov.io access token.")
            return self._access_token

        except requests.RequestException as e:
            logger.error(f"Error requesting Snov.io access token: {e}")
            raise SnovError(
                f"Could not connect to Snov.io to get access token: {e}"
            ) from e

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        A helper method to make authenticated requests to the Snov.io API.

        Args:
            method (str): HTTP method (e.g., 'GET', 'POST').
            endpoint (str): API endpoint path (e.g., '/v1/get-emails-from-url').
            **kwargs: Additional arguments to pass to requests (params, data, json).

        Returns:
            Dict[str, Any]: The JSON response from the API.
        """
        token = self._get_access_token()
        url = f"{self.BASE_URL}{endpoint}"

        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        kwargs["headers"] = headers

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

    def find_emails_by_name_and_domain(
        self, first_name: str, last_name: str, domain: str
    ) -> Dict[str, Any]:
        """
        Starts a task to find an email by name and domain.
        Note: This is an asynchronous operation on Snov.io's side.

        Returns:
            Dict[str, Any]: The initial response containing the task_hash.
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

        Args:
            task_hash (str): The hash of the task initiated by `find_emails_by_name_and_domain`.

        Returns:
            Dict[str, Any]: The result of the email search task.
        """
        params = {"task_hash": task_hash}
        return self._make_request(
            "GET", "/v2/emails-by-domain-by-name/result", params=params
        )

    def get_profile_by_url(self, profile_url: str) -> Dict[str, Any]:
        """
        Fetches a prospect's profile from Snov.io using their LinkedIn URL.
        This combines the 'add-url-for-search' and 'get-emails-from-url' steps.

        Args:
            profile_url (str): The public LinkedIn profile URL.

        Returns:
            Dict[str, Any]: The enriched profile data from Snov.io.
        """
        # Step 1: Add the URL to the processing queue
        # Snov.io's V1 API uses form data instead of JSON
        add_url_data = {"url": profile_url}
        self._make_request("POST", "/v1/add-url-for-search", data=add_url_data)

        # Snov.io needs time to process. A delay is necessary.
        # In a real task, you might poll instead of sleeping.
        time.sleep(15)

        # Step 2: Retrieve the processed data
        get_emails_data = {"url": profile_url}
        return self._make_request(
            "POST", "/v1/get-emails-from-url", data=get_emails_data
        )
