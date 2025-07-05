"""
This script creates the Streamlit user interface for the Lead Generation tool.
Its primary role is to collect user input, trigger the backend Celery pipeline,
and display the status of recent pipeline runs.
"""

import os
import sys
import logging
import uuid
import pandas as pd
import streamlit as st
from pymongo import MongoClient, DESCENDING
from datetime import datetime

# --- Project Path Setup ---
# This ensures that the app can import other project modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Local Imports ---
from config import MONGO_URI, MONGO_DB_NAME, SNOV_CLIENT_ID, SNOV_CLIENT_SECRET
from tasks import start_lead_generation_pipeline
from clients.snov_client import SnovClient, SnovError

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(page_title="LeadGen Pipeline", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Lead Generation Pipeline 🚀</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")


# --- Cached Functions for Data Fetching ---
@st.cache_resource
def get_mongo_client():
    """Cached function to get a MongoDB client."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ismaster")
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None


@st.cache_resource
def get_snov_client():
    """Cached function to get a Snov.io client."""
    if not SNOV_CLIENT_ID or not SNOV_CLIENT_SECRET:
        st.error("Snov.io API credentials are not configured.")
        return None
    try:
        return SnovClient(client_id=SNOV_CLIENT_ID, client_secret=SNOV_CLIENT_SECRET)
    except Exception as e:
        st.error(f"Failed to initialize Snov.io client: {e}")
        return None


client = get_mongo_client()
snov_client = get_snov_client()


@st.cache_data(ttl=600)
def get_dropdown_data(_client, collection_name):
    """Cached function to fetch dropdown data from MongoDB."""
    if not _client:
        return ["-- DB Connection Failed --"], {}
    try:
        collection = _client[MONGO_DB_NAME][collection_name]
        data = list(collection.find({}, {"_id": 0, "ID": 1, "Description": 1}))
        desc_to_id = {item["Description"]: item["ID"] for item in data}
        options = ["-- None --"] + sorted(list(desc_to_id.keys()))
        return options, desc_to_id
    except Exception as e:
        logger.error(f"Failed to fetch data for {collection_name}: {e}")
        return ["-- Error --"], {}


@st.cache_data(ttl=300)
def get_snov_lists(_snov_client):
    """Cached function to fetch Snov.io lists."""
    if not _snov_client:
        return {}
    try:
        existing_lists = _snov_client.get_user_lists()
        return {item["name"]: item["id"] for item in existing_lists}
    except SnovError as e:
        st.warning(f"Could not fetch Snov.io lists: {e}")
        return {}


# --- UI Layout and Input Fields ---

# Fetch data for dropdowns
industry_options, industry_map = get_dropdown_data(client, "IndustryCodesV2")
function_options, function_map = get_dropdown_data(client, "JobFunctionID")
location_options, location_map = get_dropdown_data(client, "LocationID")
list_name_to_id = get_snov_lists(snov_client)

st.markdown("### 1. Define Job Search Criteria")
col1, col2, col3 = st.columns(3)

with col1:
    keywords = st.text_input("🔍 Job Title Keywords (comma-separated)")
    job_type = st.selectbox(
        "🧰 Job Type", ["-- None --", "fullTime", "partTime", "contract", "internship"]
    )

with col2:
    date_posted = st.selectbox(
        "📅 Date Posted", ["pastMonth", "anyTime", "pastWeek", "past24Hours"]
    )
    onsite_remote = st.selectbox(
        "🏢 Onsite/Remote", ["-- None --", "onSite", "remote", "hybrid"]
    )

with col3:
    st.markdown("📄 Snov.io Prospect List")
    dropdown_options = sorted(list(list_name_to_id.keys())) + ["➕ Create new list"]
    selected_option = st.selectbox(
        "Select or create a list", dropdown_options, label_visibility="collapsed"
    )

    snov_list_id = st.session_state.get("snov_list_id")
    if selected_option == "➕ Create new list":
        new_list_name = st.text_input("Enter name for new list")
        if new_list_name and snov_client:
            if st.button("Create and Use This List"):
                try:
                    response = snov_client.create_prospect_list(name=new_list_name)
                    if response.get("success"):
                        st.session_state.snov_list_id = response.get("id")
                        st.success(
                            f"Created new list: {new_list_name} (ID: {st.session_state.snov_list_id})"
                        )
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("Failed to create new list.")
                except SnovError as e:
                    st.error(f"Error creating list: {e}")
    else:
        st.session_state.snov_list_id = list_name_to_id.get(selected_option)

st.markdown("### 2. Add Granular Filters")
col4, col5, col6 = st.columns(3)
with col4:
    selected_location = st.selectbox("🌐 Location", location_options)
with col5:
    selected_industry = st.selectbox("🏭 Industry", industry_options)
with col6:
    selected_function = st.selectbox("💼 Job Function", function_options)

st.markdown("---")

# --- Search Button and Pipeline Trigger ---
if st.button(
    "🚀 Start Lead Generation Pipeline", use_container_width=True, type="primary"
):
    user_email = st.user.email if hasattr(st.user, "email") else "not_logged_in"

    if not keywords:
        st.warning("Please enter at least one job title keyword.")
    elif not st.session_state.snov_list_id:
        st.warning("Please select or create a Snov.io prospect list.")
    else:
        with st.spinner("Dispatching job to the processing pipeline..."):
            location_id = location_map.get(selected_location)
            industry_id = industry_map.get(selected_industry)
            function_id = function_map.get(selected_function)

            pipeline_params = {
                "search_criteria": {
                    "keywords": keywords,
                    "location": location_id,
                    "date_posted": date_posted,
                    "job_type": job_type if job_type != "-- None --" else None,
                    "function_id": function_id,
                    "industry_id": industry_id,
                    "onsite_remote": onsite_remote
                    if onsite_remote != "-- None --"
                    else None,
                    "sort": "mostRecent",
                },
                "snov_list_id": st.session_state.snov_list_id,
                "run_id": str(uuid.uuid4()),
            }

            try:
                start_lead_generation_pipeline.delay(pipeline_params)

                if client:
                    query_log = {
                        "run_id": pipeline_params["run_id"],
                        "criteria_keywords": pipeline_params["search_criteria"][
                            "keywords"
                        ],
                        "submitted_by": user_email,
                        "current_step": "Pipeline Queued",
                        "submitted_at": datetime.utcnow(),
                    }
                    db = client[MONGO_DB_NAME]
                    db.Query.update_one(
                        {"run_id": query_log["run_id"]},
                        {"$set": query_log},
                        upsert=True,
                    )

                st.success(
                    f"✅ Pipeline dispatched successfully! Run ID: {pipeline_params['run_id']}"
                )
                st.info(
                    "The process is running in the background. See the dashboard below for live status updates."
                )

            except Exception as e:
                st.error(
                    f"❌ Failed to start the pipeline. Could not connect to the task queue."
                )
                logger.error(f"Streamlit UI failed to dispatch Celery task: {e}")

st.markdown("---")
st.markdown("### 📊 Pipeline Status Dashboard")

if st.button("🔄 Refresh Status"):
    st.rerun()

if client:
    try:
        db = client[MONGO_DB_NAME]
        recent_queries = list(
            db.Query.find().sort("submitted_at", DESCENDING).limit(10)
        )

        if not recent_queries:
            st.info("No recent pipeline runs found.")
        else:
            for query in recent_queries:
                status = query.get("current_step", "UNKNOWN")
                submitted_by = query.get("submitted_by", "N/A")
                keywords = query.get("criteria_keywords", "N/A")

                status_emoji = "⏳"
                if "Completed" in status or "Finished" in status:
                    status_emoji = "✅"
                elif "Failed" in status or "Error" in status:
                    status_emoji = "❌"

                with st.expander(
                    f"{status_emoji} **{keywords}** (Submitted by: {submitted_by})"
                ):
                    st.info(f"**Current Step:** {status}")
                    st.text(f"Run ID: {query.get('run_id')}")
                    st.text(
                        f"Last Update: {query.get('last_updated', query.get('submitted_at')).strftime('%Y-%m-%d %H:%M:%S')} UTC"
                    )

    except Exception as e:
        st.error(f"Could not fetch pipeline statuses from the database: {e}")
else:
    st.warning("Cannot display pipeline status. No connection to MongoDB.")
