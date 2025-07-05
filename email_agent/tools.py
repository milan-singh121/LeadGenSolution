"""
This script defines the tools available to the email generation agent,
including database access and web search capabilities.
"""

import asyncio
import os
import sys
from typing import Dict
from datetime import datetime
from langchain_core.tools import tool
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# Import the database collections from the refactored db_client
from email_agent.db_client import (
    people_collection,
    posts_collection,
    about_company_collection,
    blueprints_collection,
    generated_emails_collection,
    clients_collection,
)

app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir))  # already project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Import API keys from the central config file
from config import OPENAI_API_KEY

# --- Tool Initializations ---
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

embedding_client = OpenAI(api_key=OPENAI_API_KEY)


@tool
def create_text_embedding(text: str) -> list[float]:
    """
    Creates a vector embedding for a given text using OpenAI's text-embedding-3-small model.
    """
    print(f"---TOOL: Creating embedding for text: '{text}'---")
    try:
        response = embedding_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return []


def is_valid_vector(vec):
    return np.isfinite(vec).all() and np.linalg.norm(vec) > 0


@tool
async def find_similar_clients(embedding: list[float]) -> list[dict]:
    """
    Finds the most similar client to the given embedding using cosine similarity.
    """
    print("---TOOL: Performing manual vector search for similar clients---")
    if not embedding:
        return [{"error": "No embedding provided for search."}]

    try:
        query_embedding = np.array(embedding).reshape(1, -1)
        if not is_valid_vector(query_embedding):
            return [{"error": "Query embedding is invalid (NaN/Inf or zero vector)."}]

        # ✅ Correct async cursor handling
        all_docs = await clients_collection.find(
            {"embedding": {"$exists": True}},
            {"_id": 0, "message_date": 0},
        ).to_list(length=None)

        if not all_docs:
            return [{"error": "No client embeddings found in database."}]

        # Calculate cosine similarity
        scored_docs = []
        for doc in all_docs:
            try:
                doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
                if not is_valid_vector(doc_embedding):
                    continue
                score = cosine_similarity(query_embedding, doc_embedding)[0][0]
                doc["similarity_score"] = round(float(score), 4)
                scored_docs.append(doc)
            except Exception as e:
                print(f"Error comparing embedding: {e}")
                continue

        # Sort by similarity score and return top 1
        scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)

        if scored_docs:
            top_match = scored_docs[0]
            top_match.pop("embedding", None)
            top_match.pop("similarity_score", None)
            return [top_match]
        else:
            return [{"error": "No similar clients found."}]

    except Exception as e:
        return [{"error": f"Vector comparison error: {e}"}]


@tool
async def get_prospect_info(linkedin_url: str) -> dict:
    """
    Retrieves detailed information about a prospect from the database using their LinkedIn profile URL.
    """
    print(f"---TOOL: Searching for prospect by URL: {linkedin_url}---")
    try:
        document = await people_collection.find_one(
            {"profileURL": linkedin_url}, {"_id": 0, "message_date": 0}
        )
        if document and "_id" in document:
            document["_id"] = str(document["_id"])
        return document if document else {"error": "Prospect not found."}
    except Exception as e:
        return {"error": f"Database error: {e}"}


@tool
async def get_prospect_posts(username: str) -> list:
    """
    Retrieves the most recent LinkedIn posts for a specific prospect using their username.
    """
    print(f"---TOOL: Searching for posts by username: {username}---")
    try:
        cursor = (
            posts_collection.find({"username": username}, {"_id": 0, "message_date": 0})
            .sort("postedDate", -1)
            .limit(3)
        )
        posts = await cursor.to_list(length=3)
        for post in posts:
            if "_id" in post:
                post["_id"] = str(post["_id"])
            if "text" in post:
                post["post_content"] = post.pop("text")
        return posts
    except Exception as e:
        return [{"error": f"Database error: {e}"}]


@tool
async def get_company_info() -> dict:
    """
    Retrieves information about our company (the sender) to include in the email.
    """
    print("---TOOL: Retrieving our company info---")
    try:
        document = await about_company_collection.find_one(
            {}, {"_id": 0, "message_date": 0}
        )
        if document and "_id" in document:
            document["_id"] = str(document["_id"])
        return document if document else {"error": "Company info not found."}
    except Exception as e:
        return {"error": f"Database error: {e}"}


@tool
async def get_email_blueprint(sequence_id: int = 1) -> dict:
    """
    Retrieves the email structure, guidelines, and template for a given step in the sequence.
    """
    print(f"---TOOL: Retrieving blueprint for sequence ID: {sequence_id}---")
    try:
        document = await blueprints_collection.find_one(
            {"sequence_id": sequence_id}, {"_id": 0, "message_date": 0}
        )
        if document and "_id" in document:
            document["_id"] = str(document["_id"])
        return document if document else {"error": "Blueprint not found."}
    except Exception as e:
        return {"error": f"Database error: {e}"}


@tool
async def save_email_sequence(
    linkedin_url: str, emails: Dict, total_cost: float
) -> dict:
    """
    Saves or updates the full email sequence for a prospect in a single document.
    """
    print(f"---TOOL: Saving full email sequence for {linkedin_url}---")
    try:
        update_data = {
            "linkedin_url": linkedin_url,
            "total_cost": total_cost,
            "message_date": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        }
        update_data.update(emails)

        result = await generated_emails_collection.update_one(
            {"linkedin_url": linkedin_url}, {"$set": update_data}, upsert=True
        )
        return {
            "status": "success",
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id),
        }
    except Exception as e:
        return {"status": "error", "message": f"Database error: {e}"}
