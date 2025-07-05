"""
This script provides a refactored client to interact with a LinkedIn Data API
via RapidAPI. It is designed for use in a decoupled, task-based architecture.
"""

import os
import time
import random
import logging
from typing import Dict, Any, Optional, List

import requests

# It's good practice to have a logger configured for your module.
# In a real application, you would configure this logger in your app's entry point.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RapidAPIError(Exception):
    """Custom exception for RapidAPI client errors."""

    pass


class RapidAPIClient:
    """
    A client to interact with a LinkedIn Data API via RapidAPI.

    This class is designed to be instantiated by a worker process. It is not a
    singleton and relies on its dependencies (API key, base URL) being injected
    during initialization. This makes it more flexible and testable.
    """

    def __init__(
        self, api_key: str, base_url: str = "https://linkedin-api8.p.rapidapi.com"
    ) -> None:
        """
        Initializes the RapidAPI client.

        Args:
            api_key (str): Your RapidAPI key.
            base_url (str): The base URL for the API.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")

        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key

        try:
            self.host = self.base_url.split("://")[1]
        except IndexError:
            raise ValueError(
                f"Invalid base_url format: {base_url}. It should include a scheme (e.g., https://)."
            )

        self.headers: Dict[str, str] = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host,
            "Content-Type": "application/json",
        }

    def _make_request(
        self, endpoint: str, params: Dict[str, Any], delay_seconds: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method to perform a GET request to a specific API endpoint.
        """
        filtered_params = {k: v for k, v in params.items() if v is not None}
        url = f"{self.base_url}{endpoint}"
        time.sleep(delay_seconds + random.uniform(0.5, 2.0))

        try:
            logger.info(f"Making GET request to {url} with params: {filtered_params}")
            response = requests.get(
                url, headers=self.headers, params=filtered_params, timeout=45
            )
            response.raise_for_status()
            if not response.content:
                logger.warning(f"Empty response received from {url}")
                return {}
            return response.json()
        except requests.HTTPError as e:
            error_text = e.response.text if e.response else "No response body"
            logger.error(f"HTTP Error for {url}: {e} | Response: {error_text}")
            raise RapidAPIError(
                f"HTTP Error: {e.response.status_code} for {url} - {error_text}"
            ) from e
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Request or JSON decode error for {url}: {e}")
            raise RapidAPIError(f"Request/JSON error for {url}: {e}") from e

    def search_jobs(
        self,
        keywords: Optional[str] = None,
        location: Optional[str] = None,
        job_type: Optional[str] = None,
        function_id: Optional[str] = None,
        industry_id: Optional[str] = None,
        onsite_remote: Optional[str] = None,
        start: Optional[str] = None,
        sort: str = "mostRecent",
        date_posted: str = "pastMonth",
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Search job postings with pagination to retrieve a specified number of results.
        """
        all_jobs = []
        page_size = 25
        max_start = 975

        params = {
            "keywords": keywords,
            "locationId": location,
            "datePosted": date_posted,
            "jobType": job_type,
            "functionIds": function_id,
            "industryIds": industry_id,
            "onsiteRemote": onsite_remote,
            "sort": sort,
            "start": start,
        }
        logger.info(f"Fetching jobs with params: {params}")
        logger.info(f"Limit Set: {limit}")

        for start in range(0, min(limit, max_start + page_size), page_size):
            logger.info(f"Start : {start}")
            if len(all_jobs) >= limit:
                break

            page_params = params.copy()
            page_params["start"] = str(start)

            try:
                response_data = self._make_request("/search-jobs", page_params)

                # --- FIX: Handle multiple possible response structures ---
                jobs_on_page = []
                if isinstance(response_data, list):
                    jobs_on_page = response_data
                elif isinstance(response_data, dict):
                    # Check for common keys like 'data', 'results', or 'jobs'
                    for key in ["data", "results", "jobs"]:
                        if key in response_data and isinstance(
                            response_data[key], list
                        ):
                            jobs_on_page = response_data[key]
                            break

                if not jobs_on_page:
                    logger.info(
                        f"No jobs found on page starting at {start}. Stopping pagination."
                    )
                    break

                all_jobs.extend(jobs_on_page)
                logger.info(
                    f"Fetched {len(jobs_on_page)} jobs. Total jobs so far: {len(all_jobs)}"
                )

            except RapidAPIError as e:
                logger.error(f"Failed to fetch jobs page at start={start}: {e}")
                break

        return all_jobs[:limit]

    def get_hiring_team(self, job_id: str, job_url: str) -> Optional[Dict[str, Any]]:
        """Retrieve hiring team information for a specific job."""
        return self._make_request("/get-hiring-team", {"id": job_id, "url": job_url})

    def search_people(self, keywords: str, company: str) -> Optional[Dict[str, Any]]:
        """Search LinkedIn people by keywords and current company."""
        return self._make_request(
            "/search-people",
            {"keywords": keywords, "start": "0", "company": company},
        )

    def get_linkedin_profile_data_by_url(
        self, profile_url: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch LinkedIn profile data using a public profile URL."""
        return self._make_request("/get-profile-data-by-url", {"url": profile_url})

    def get_linkedin_posts_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get the last 50 LinkedIn posts from a user by username."""
        return self._make_request("/get-profile-posts", {"username": username})

    def get_linkedin_profile_comments(self, username: str) -> Optional[Dict[str, Any]]:
        """Get the last 50 comments made by a LinkedIn user."""
        return self._make_request("/get-profile-comments", {"username": username})

    def get_linkedin_company_details_by_id(
        self, company_id: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch company details using a LinkedIn Company ID."""
        return self._make_request("/get-company-details-by-id", {"id": company_id})

    def get_linkedin_company_details_by_username(
        self, username: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch company details using a LinkedIn Company Username."""
        return self._make_request("/get-company-details", {"username": username})

    def get_company_posts_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch company linkedin posts using a LinkedIn Company Username."""
        return self._make_request("/get-company-posts", {"username": username})
