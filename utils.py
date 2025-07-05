"""
This script contains a collection of helper functions for processing and cleaning
LinkedIn and other data sources, often retrieved in string or complex
dictionary/list formats. These are designed to be used by the Celery tasks.
"""

import ast
import json
import re
import logging
import pandas as pd
from datetime import datetime
import tldextract

logger = logging.getLogger(__name__)


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def find_relevant_experience(experiences, target_company=None):
    if isinstance(experiences, str):
        try:
            experiences = ast.literal_eval(experiences)
        except Exception:
            logging.warning(f"Could not parse stringified experiences: {experiences}")
            return None

    if not isinstance(experiences, list):
        logging.warning(f"Unexpected format for experiences: {experiences}")
        return None

    # 1. Prefer match on companyName
    if target_company:
        for val in experiences:
            if (
                isinstance(val, dict)
                and val.get("companyName", "").strip().lower()
                == target_company.strip().lower()
            ):
                return val

    # 2. Return first current role (no end date)
    for val in experiences:
        if not isinstance(val, dict):
            continue
        end_date = val.get("end", {})
        if all(end_date.get(k, 0) in [0, None] for k in ("year", "month", "day")):
            return val

    # 3. Fallback to first experience
    return experiences[0] if experiences else None


def process_full_positions(entry):
    if isinstance(entry, str):
        try:
            entry = ast.literal_eval(entry)
        except (ValueError, SyntaxError):
            return []  # or log error, depending on use case
    elif not isinstance(entry, list):
        return []

    return [
        {
            "companyId": p.get("companyId", ""),
            "companyName": p.get("companyName", ""),
            "companyUsername": p.get("companyUsername", ""),
            "companyURL": p.get("companyURL", ""),
            "companyLogo": p.get("companyLogo", ""),
            "companyIndustry": p.get("companyIndustry", ""),
            "companyStaffCountRange": p.get("companyStaffCountRange", ""),
            "title": p.get("title", ""),
            "location": p.get("location", ""),
            "description": p.get("description", ""),
            "employmentType": p.get("employmentType", ""),
            "start": p.get("start", {}),
            "end": p.get("end", {}),
        }
        for p in entry
    ]


def process_skills(entry):
    # Return empty list for NaN or None
    if entry is None or isinstance(entry, float) and pd.isna(entry):
        return []

    # If it's a string (e.g., from JSON or stringified list), try to parse
    if isinstance(entry, str):
        try:
            entry = ast.literal_eval(entry)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string entry: {entry} -> {e}")
            return []

    # If it's already a list of dicts, extract 'name'
    if isinstance(entry, list):
        return [item.get("name", "") for item in entry if isinstance(item, dict)]

    # Fallback
    return []


def parse_llm_json_response(response_text: str):
    """Safely parses a JSON object or list from a string that might contain surrounding text."""
    # Find the start of the JSON array or object
    start_bracket = response_text.find("[")
    start_brace = response_text.find("{")

    if start_bracket == -1 and start_brace == -1:
        return []

    if start_bracket != -1 and (start_bracket < start_brace or start_brace == -1):
        start = start_bracket
        end_char = "]"
    else:
        start = start_brace
        end_char = "}"

    end = response_text.rfind(end_char)
    if end == -1:
        return []

    json_like = response_text[start : end + 1]

    try:
        return json.loads(json_like)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        return []


def clean_email(value):
    """Cleans an email field, returning the first email from a list if applicable."""
    if isinstance(value, list) and value:
        return value[0]
    elif isinstance(value, str):
        return value.strip()
    return None


def format_final_data(final_data_df: pd.DataFrame):
    """Restructures a DataFrame to nest questionnaire and email data for final output."""

    def extract_questions(row, fields):
        return {f"question_{i + 1}": fields[i] for i in range(len(fields))}

    def extract_answers(row, fields):
        return {f"answer_{i + 1}": row.get(fields[i], "") for i in range(len(fields))}

    question_cols = [col for col in final_data_df.columns if "?" in col]
    email_cols = [
        col for col in final_data_df.columns if "Subject" in col or "Email Body" in col
    ]

    if question_cols:
        final_data_df["questions"] = final_data_df.apply(
            lambda row: extract_questions(row, question_cols), axis=1
        )
        final_data_df["answers"] = final_data_df.apply(
            lambda row: extract_answers(row, question_cols), axis=1
        )
        final_data_df.drop(columns=question_cols, inplace=True)

    if email_cols:
        final_data_df["email_data"] = final_data_df.apply(
            lambda row: {col: row[col] for col in email_cols}, axis=1
        )
        final_data_df.drop(columns=email_cols, inplace=True)

    return final_data_df.to_dict(orient="records")


def flatten_person_data(data):
    flat = {}

    # These nested keys will be expanded separately
    nested_keys = ["currentJob", "previousJob", "social"]

    # Include all top-level keys except nested ones we handle separately
    for key, value in data.items():
        if key not in nested_keys:
            flat[key] = value if value is not None else None  # Explicitly preserve None

    # Define expected sub-keys for currentJob and social
    current_job_keys = [
        "companyName",
        "position",
        "socialLink",
        "site",
        "locality",
        "state",
        "city",
        "street",
        "street2",
        "postal",
        "founded",
        "startDate",
        "endDate",
        "size",
        "industry",
        "companyType",
        "country",
    ]
    social_keys = ["link", "source"]

    # Flatten currentJob[0] if exists
    current_job = data.get("currentJob", [{}])
    job_data = current_job[0] if current_job else {}
    for key in current_job_keys:
        flat[f"currentJob_{key}"] = job_data.get(key, None)

    # Flatten social[0] if exists
    social = data.get("social", [{}])
    social_data = social[0] if social else {}
    for key in social_keys:
        flat[f"social_{key}"] = social_data.get(key, None)

    # Ensure 'emails' and 'previousJob' are explicitly preserved
    flat["emails"] = data.get("emails", None)
    flat["previousJob"] = data.get("previousJob", None)

    return flat


def extract_emails(email_data):
    if not email_data:
        return None  # or return '' if you prefer empty string

    emails = [item["email"] for item in email_data if "email" in item]
    return ", ".join(emails) if emails else None


def extract_domain(url):
    if not isinstance(url, str) or not url:
        return None
    extracted = tldextract.extract(url)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return None


def process_post_author(entry):
    # If the entry is NaN or None
    if entry is None or isinstance(entry, float) and pd.isna(entry):
        return []

    # If it's a string, attempt to parse it
    if isinstance(entry, str):
        try:
            entry = ast.literal_eval(entry)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing author entry: {entry} -> {e}")
            return []

    # If it's a single dict (not a list), convert to a list
    if isinstance(entry, dict):
        entry = [entry]

    # If it's not a list at this point, return empty
    if not isinstance(entry, list):
        return []

    # Process the list of authors
    processed_author = [
        {
            "firstName": p.get("firstName", ""),
            "lastName": p.get("lastName", ""),
            "headline": p.get("headline", ""),
            "username": p.get("username", ""),
            "url": p.get("url", ""),
        }
        for p in entry
        if isinstance(p, dict)
    ]
    return processed_author


def process_reshared_data(entry):
    # If the entry is NaN or None
    if entry is None or isinstance(entry, float) and pd.isna(entry):
        return []

    # If it's a string, attempt to parse it
    if isinstance(entry, str):
        try:
            entry = ast.literal_eval(entry)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing author entry: {entry} -> {e}")
            return []

    # If it's a single dict (not a list), convert to a list
    if isinstance(entry, dict):
        entry = [entry]

    # If it's not a list at this point, return empty
    if not isinstance(entry, list):
        return []

    # Process the list of authors
    processed_reshared = [
        {
            "text": p.get("text", ""),
            "postUrl": p.get("postUrl", ""),
            "author": p.get("author", ""),
            "author_url": p.get("url", ""),
            "url": p.get("url", ""),
        }
        for p in entry
        if isinstance(p, dict)
    ]
    return processed_reshared


def clean_data_for_questionnaire(final_people_data, people_posts):
    # Clean People's Data
    people_dict = {
        f"Prospect {i + 1}": row
        for i, (_, row) in enumerate(
            final_people_data.drop(
                columns="processed_fullPositions", errors="ignore"
            ).iterrows()
        )
    }

    # Merge name info into posts
    name_cols = ["username", "firstName", "lastName"]
    people_posts = pd.merge(
        people_posts, final_people_data[name_cols], on="username", how="left"
    )

    # Clean and filter post dates
    people_posts["postedDate"] = pd.to_datetime(
        people_posts["postedDate"].str.replace(" +0000 UTC", "", regex=False),
        errors="coerce",
    )
    cutoff_date = datetime.utcnow() - pd.DateOffset(months=2)

    # Filter recent, original posts
    recent_posts = people_posts[
        (people_posts["postedDate"] >= cutoff_date) & (~people_posts["repost"])
    ]

    return people_dict, recent_posts
