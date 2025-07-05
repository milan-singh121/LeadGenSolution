"""
This script contains utility functions for the email agent,
specifically for parsing XML responses from the LLM.
"""

import re
from typing import List, Dict


def extract_between_tags(tag: str, string: str, strip: bool = False) -> List[str]:
    """
    Extracts content between XML-like tags using regex.
    """
    ext_list = re.findall(f"<{tag}>(.*?)</{tag}>", string, re.DOTALL)
    if strip:
        return [e.strip() for e in ext_list]
    return ext_list


def extract_email_details(email_content: str) -> Dict[str, str]:
    """
    Extracts details from a single <email> tag.
    """
    sequence = extract_between_tags("sequence", email_content, strip=True)
    subject = extract_between_tags("subject", email_content, strip=True)
    body_raw = extract_between_tags("body", email_content, strip=True)

    body_text = body_raw[0].strip() if body_raw else ""
    cleaned_body = body_text.replace("\\n", " ").replace("\n", " ")
    cleaned_body = re.sub(r"\s+", " ", cleaned_body).strip()

    return {
        "sequence": sequence[0] if sequence else "",
        "subject": subject[0] if subject else "",
        "body": cleaned_body,
    }


def extract_emails_from_llm_response(xml_string: str) -> List[Dict[str, str]]:
    """
    Extracts all email details from a root <emails> tag in the LLM's response.
    """
    # First, find the content within the main <emails> tag
    emails_content_list = extract_between_tags("emails", xml_string, strip=True)
    if not emails_content_list:
        return []

    # Then, find all individual <email> tags within that content
    email_contents = extract_between_tags("email", emails_content_list[0], strip=True)

    # Parse each individual email content
    return [extract_email_details(content) for content in email_contents]
