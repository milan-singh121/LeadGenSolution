"""
This file contains the complete, end-to-end Celery tasks for the lead
generation pipeline. Each function represents a distinct, retry-able step
in the workflow, incorporating all logic from the original pipeline scripts.
"""

import time
import os
import json
import logging
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from celery import chain
from pymongo import MongoClient, UpdateOne
from typing import Dict, Any, Optional, List, Set

from clients.rapid_api_client import RapidAPIClient, RapidAPIError
from clients.snov_client import SnovClient, SnovError
from clients.llm_clients import ClaudeClient
from question_prompt import get_question_summarizer_prompts
from celery_app import celery_app

from config import (
    RAPID_API_KEY,
    MONGO_URI,
    MONGO_DB_NAME,
    SNOV_CLIENT_ID,
    SNOV_CLIENT_SECRET,
    BLOCKED_INDUSTRIES,
    PROSPECT_SEARCH_KEYWORDS,
)
from email_agent.main import generate_email_sequence
from utils import (
    flatten_dict,
    parse_llm_json_response,
    process_full_positions,
    find_relevant_experience,
    process_skills,
    extract_domain,
    flatten_person_data,
    extract_emails,
    process_post_author,
    process_reshared_data,
    clean_data_for_questionnaire,
)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper function to update pipeline status ---
def update_status(run_id: str, message: str):
    """Updates the current step for a given run_id in the Query collection."""
    if not run_id:
        return
    try:
        db = MongoClient(MONGO_URI)[MONGO_DB_NAME]
        db.Query.update_one(
            {"run_id": run_id},
            {"$set": {"current_step": message, "last_updated": datetime.utcnow()}},
        )
        logger.info(f"STATUS UPDATE for {run_id}: {message}")
    except Exception as e:
        logger.error(f"Failed to update status for run_id {run_id}: {e}")


# --- Task Definitions ---


@celery_app.task(
    bind=True,
    autoretry_for=(RapidAPIError,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
)
def fetch_and_clean_jobs_task(self, pipeline_params: dict, start_offset: int = 0):
    """
    Task 1: Fetches new jobs, cleans them, and saves them to MongoDB.
    """
    run_id = pipeline_params.get("run_id")
    search_criteria = pipeline_params.get("search_criteria", {})
    update_status(run_id, f"Fetching Jobs (Batch starting at {start_offset})...")

    logger.info(
        f"TASK 1: Starting job fetch with criteria: {search_criteria} and offset: {start_offset}"
    )
    api_client = RapidAPIClient(api_key=RAPID_API_KEY)
    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]

    jobs_raw = api_client.search_jobs(**search_criteria, start=start_offset)
    if not jobs_raw:
        logger.warning("No jobs found from API.")
        return []

    jobs_df = pd.DataFrame([flatten_dict(job) for job in jobs_raw])
    jobs_df.rename(columns={"id": "job_id"}, inplace=True)
    jobs_df = jobs_df.dropna(subset=["company_url"]).reset_index(drop=True)
    jobs_df["company_username"] = jobs_df["company_url"].str.extract(
        r"/company/([^/]+)/"
    )
    jobs_df = jobs_df.dropna(subset=["company_username"])

    existing_job_ids = {doc["job_id"] for doc in db.RawJobs.find({}, {"job_id": 1})}
    new_jobs_df = jobs_df[~jobs_df["job_id"].isin(existing_job_ids)].copy()

    if new_jobs_df.empty:
        logger.info("No new jobs to process after filtering existing ones.")
        return []

    # Add message_date timestamp
    timestamp = datetime.utcnow()
    new_jobs_records = new_jobs_df.to_dict("records")
    for record in new_jobs_records:
        record["message_date"] = timestamp
        db.RawJobs.update_one(
            {"job_id": record["job_id"]},
            {"$set": record},
            upsert=True,
        )

    jobs_to_clean = new_jobs_df.copy()
    jobs_to_clean.rename(columns={"url": "job_url"}, inplace=True)
    cols_to_drop = ["referenceId", "posterId", "company_logo", "postedTimestamp"]
    jobs_to_clean.drop(
        columns=[c for c in cols_to_drop if c in jobs_to_clean.columns], inplace=True
    )

    cleaned_records = jobs_to_clean.to_dict("records")
    for record in cleaned_records:
        record["message_date"] = timestamp
        db.Jobs.update_one(
            {"job_id": record["job_id"]},
            {"$set": record},
            upsert=True,
        )

    logger.info(f"TASK 1: Successfully processed and saved {len(new_jobs_df)} jobs.")
    return new_jobs_df["job_id"].tolist()


@celery_app.task(
    bind=True,
    autoretry_for=(RapidAPIError,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
)
def process_and_filter_companies_task(self, new_job_ids: list, pipeline_params: dict):
    """
    Task 2: Fetches company details for new jobs, cleans the data,
    filters them based on ICP criteria, and saves the results.
    """
    run_id = pipeline_params.get("run_id")
    update_status(run_id, "Processing and Filtering Companies...")

    if not new_job_ids:
        logger.info("TASK 2: No new job IDs received. Skipping.")
        return []

    logger.info(f"TASK 2: Processing companies for {len(new_job_ids)} jobs.")
    api_client = RapidAPIClient(api_key=RAPID_API_KEY)
    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]

    # 1. Get company usernames associated with the new jobs
    jobs_cursor = db.Jobs.find(
        {"job_id": {"$in": new_job_ids}},
        {"company_username": 1, "job_id": 1, "_id": 0},
    )
    # Create a map of username -> list of job_ids
    username_to_job_ids = {}
    for job in jobs_cursor:
        username = job.get("company_username")
        if username:
            username_to_job_ids.setdefault(username, []).append(job["job_id"])

    if not username_to_job_ids:
        logger.warning("No company usernames found for the given job IDs.")
        return []

    # 2. Find which companies are missing from our RawCompany collection
    all_usernames = list(username_to_job_ids.keys())
    existing_companies_cursor = db.RawCompany.find(
        {"universalName": {"$in": all_usernames}}, {"universalName": 1}
    )
    existing_usernames = {doc["universalName"] for doc in existing_companies_cursor}
    missing_usernames = set(all_usernames) - existing_usernames

    # 3. Fetch data for missing companies
    newly_fetched_companies = []
    if missing_usernames:
        logger.info(f"Fetching data for {len(missing_usernames)} new companies.")
        for username in missing_usernames:
            try:
                company_data = api_client.get_linkedin_company_details_by_username(
                    username
                )
                if company_data:
                    company_data = company_data.get("data", {})
                    if isinstance(company_data, dict):
                        company_data["companyId"] = company_data.pop("id", None)
                        newly_fetched_companies.append(company_data)
            except RapidAPIError as e:
                logger.error(f"Could not fetch company {username}: {e}")

    # 4. Insert newly fetched raw data
    if newly_fetched_companies:
        operations = []
        for company in newly_fetched_companies:
            company["message_date"] = datetime.utcnow()
            company_id = company.get("companyId")
            if company_id:
                operations.append(
                    UpdateOne(
                        {"companyId": company_id},
                        {"$set": company},
                        upsert=True,
                    )
                )

        if operations:
            result = db.RawCompany.bulk_write(operations)
            logger.info(
                f"Upserted {result.upserted_count} new companies and updated {result.modified_count} existing ones in RawCompany."
            )

    # 5. Process all relevant companies (both existing and newly fetched)
    all_relevant_companies_cursor = db.RawCompany.find(
        {"universalName": {"$in": all_usernames}}
    )

    # Helper for parsing staff range
    def parse_staff_range(range_str):
        if not isinstance(range_str, str):
            return pd.Series([None, None])
        try:
            if "+" in range_str:
                return pd.Series([int(range_str.replace("+", "")), float("inf")])
            lower, upper = map(int, range_str.split("-"))
            return pd.Series([lower, upper])
        except Exception:
            return pd.Series([None, None])

    # 6. Clean and filter the company data
    icp_fit_companies = []
    for company in all_relevant_companies_cursor:
        # Clean data
        if "staffCountRange" in company:
            limits = parse_staff_range(company["staffCountRange"])
            if limits is not None:
                company["lower_limit"], company["upper_limit"] = limits
            else:
                company["lower_limit"], company["upper_limit"] = None, None

        # Filter by industry
        industries = company.get("industries", [])
        if any(ind in BLOCKED_INDUSTRIES for ind in industries):
            continue

        # Filter by ICP employee count (upper_limit <= 200)

        if not (
            company.get("upper_limit") is not None and company["upper_limit"] <= 200
        ):
            continue

        # If all filters pass, it's an ICP-fit company
        icp_fit_companies.append(company)

    if not icp_fit_companies:
        logger.info("No ICP-fit companies found after filtering.")
        return []

    # 7. Save cleaned, ICP-fit companies to the 'Company' collection
    # Use upsert to avoid duplicates if the task reruns
    for company in icp_fit_companies:
        company["message_date"] = datetime.utcnow()
        db.Company.update_one(
            {"companyId": company["companyId"]}, {"$set": company}, upsert=True
        )

    logger.info(
        f"Saved {len(icp_fit_companies)} ICP-fit companies to 'Company' collection."
    )

    # 8. Return the list of job_ids that correspond to the ICP-fit companies
    final_valid_job_ids = []
    fit_company_usernames = {c["universalName"] for c in icp_fit_companies}
    for username, job_ids in username_to_job_ids.items():
        if username in fit_company_usernames:
            final_valid_job_ids.extend(job_ids)

    logger.info(f"TASK 2: Found {len(final_valid_job_ids)} jobs from valid companies.")
    return final_valid_job_ids


@celery_app.task(bind=True)
def check_and_route_task(
    self,
    current_batch_job_ids: list,
    pipeline_params: dict,
    all_collected_job_ids: list,
    attempt: int,
):
    """
    Task 2.5 (Router): Checks if enough companies have been found.
    If yes, proceeds with the rest of the pipeline.
    If no, triggers another loop of job/company fetching.
    """
    run_id = pipeline_params.get("run_id")
    update_status(run_id, f"Checking lead count (Attempt {attempt})...")
    MIN_COMPANIES_REQUIRED = 2
    MAX_ATTEMPTS = 3

    all_job_ids = list(set(all_collected_job_ids + current_batch_job_ids))

    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]

    company_usernames = {
        job["company_username"]
        for job in db.Jobs.find(
            {"job_id": {"$in": all_job_ids}}, {"company_username": 1}
        )
        if "company_username" in job
    }
    fit_companies_count = db.Company.count_documents(
        {"universalName": {"$in": list(company_usernames)}}
    )

    logger.info(
        f"ROUTER: Attempt {attempt}. Total unique ICP companies found so far: {fit_companies_count}."
    )

    if fit_companies_count >= MIN_COMPANIES_REQUIRED or attempt >= MAX_ATTEMPTS:
        logger.info("ROUTER: Condition met. Proceeding to fetch people.")
        downstream_pipeline = chain(
            fetch_and_process_people_task.s(pipeline_params),
            fetch_posts_task.s(pipeline_params),
            find_emails_task.s(pipeline_params),
            generate_questionnaire_task.s(pipeline_params),
            generate_email_sequence_task.s(pipeline_params),
            dump_data_to_snov_task.s(pipeline_params),
        )
        downstream_pipeline.apply_async(args=(all_job_ids,))
    else:
        logger.info("ROUTER: Not enough companies. Triggering next search loop.")
        start_lead_generation_loop.delay(pipeline_params, all_job_ids, attempt + 1)


@celery_app.task(bind=True)
def start_lead_generation_loop(
    self, pipeline_params: dict, all_collected_job_ids: list = None, attempt: int = 1
):
    """
    Orchestrates a single loop of fetching jobs and companies.
    """
    if all_collected_job_ids is None:
        all_collected_job_ids = []

    JOBS_PER_ATTEMPT = 75
    start_offset = (attempt - 1) * JOBS_PER_ATTEMPT

    loop_chain = (
        fetch_and_clean_jobs_task.s(pipeline_params, start_offset=start_offset)
        | process_and_filter_companies_task.s(pipeline_params)
        | check_and_route_task.s(pipeline_params, all_collected_job_ids, attempt)
    )
    loop_chain.apply_async()


@celery_app.task(
    bind=True,
    autoretry_for=(RapidAPIError,),
    retry_kwargs={"max_retries": 3, "countdown": 45},
)
def fetch_and_process_people_task(self, valid_job_ids: list, pipeline_params: dict):
    """
    Task 3: Fetches recruiters and prospects, enriches their profiles, and saves them.
    """
    run_id = pipeline_params.get("run_id")
    update_status(run_id, "Fetching and Processing People...")
    if not valid_job_ids:
        return []

    logger.info(f"TASK 3: Fetching people for {len(valid_job_ids)} jobs.")
    api_client = RapidAPIClient(api_key=RAPID_API_KEY)
    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]

    jobs_df = pd.DataFrame(list(db.Jobs.find({"job_id": {"$in": valid_job_ids}})))
    if jobs_df.empty:
        return []

    recruiter_records = []
    for _, row in jobs_df.iterrows():
        try:
            team = api_client.get_hiring_team(row["job_id"], row["job_url"])
            if team and team.get("data").get("items"):
                for item in team.get("data").get("items"):
                    item.update(
                        {"job_id": row["job_id"], "company_name": row["company_name"]}
                    )
                    recruiter_records.append(item)
        except RapidAPIError:
            continue

    recruiter_df = pd.DataFrame(recruiter_records).rename(columns={"url": "profileURL"})

    logger.info(f"Total Recruiters Found {recruiter_df.shape[0]}")

    recruiter_job_ids = set(recruiter_df["job_id"]) if not recruiter_df.empty else set()
    remaining_jobs_df = jobs_df[~jobs_df["job_id"].isin(recruiter_job_ids)]

    people_records = []
    for _, row in remaining_jobs_df.iterrows():
        per_company_people = []

        for keyword in PROSPECT_SEARCH_KEYWORDS:
            try:
                people = api_client.search_people(
                    keywords=keyword, company=row["company_name"]
                )
                data = people.get("data")
                items = data.get("items", []) if isinstance(data, dict) else []

                if not isinstance(items, list):
                    items = []

                for item in items:
                    item["company_name"] = row["company_name"]

                per_company_people.extend(items)

                if len(per_company_people) >= 2:
                    break

            except RapidAPIError:
                continue

            if len(per_company_people) >= 2:
                break

        people_records.extend(per_company_people)

    people_df = pd.DataFrame(people_records)

    logger.info(f"Total People Found :, {people_df.shape[0]}")

    combined_df = pd.concat([recruiter_df, people_df], ignore_index=True)

    if "profileURL" in combined_df.columns:
        final_people_df = combined_df.drop_duplicates(subset=["profileURL"])
    else:
        final_people_df = combined_df

    logger.info(f"Final People Found :, {final_people_df.shape[0]}")

    if final_people_df.empty:
        return []

    enriched_profiles = []
    for url in final_people_df["profileURL"].dropna().unique():
        existing = db.RawPeople.find_one({"profileURL": url})
        if existing:
            enriched_profiles.append(existing)
            continue
        try:
            profile = api_client.get_linkedin_profile_data_by_url(url)
            if profile:
                profile["profileURL"] = url
                profile["message_date"] = datetime.utcnow()
                db.RawPeople.update_one(
                    {"username": profile["username"]}, {"$set": profile}, upsert=True
                )
                enriched_profiles.append(profile)
        except RapidAPIError:
            continue

    if not enriched_profiles:
        return []

    profiles_df = pd.DataFrame(enriched_profiles)

    selected_cols = [
        "id",
        "profileURL",
        "company_name",
        "username",
        "firstName",
        "lastName",
        "headline",
        "geo",
        "educations",
        "position",
        "fullPositions",
        "skills",
        "summary",
        "volunteering",
    ]
    final_df = profiles_df[
        [col for col in selected_cols if col in profiles_df.columns]
    ].copy()

    final_df["processed_fullPositions"] = (
        final_df["fullPositions"].dropna().apply(process_full_positions)
    )
    final_df["latest_experience"] = final_df.apply(
        lambda row: find_relevant_experience(
            row["processed_fullPositions"], row.get("company_name")
        ),
        axis=1,
    )

    # Normalize nested JSON columns safely
    final_df = pd.concat(
        [
            final_df.drop(columns=["latest_experience", "geo"], errors="ignore"),
            pd.json_normalize(final_df["latest_experience"]),
            pd.json_normalize(final_df["geo"]),
        ],
        axis=1,
    )
    final_df["clean_skills"] = final_df["skills"].apply(process_skills)
    final_df.rename(
        columns={"company_name": "company", "full": "full_address"}, inplace=True
    )

    cols_to_drop = [
        "start.year",
        "start.month",
        "start.day",
        "end.year",
        "end.month",
        "end.day",
        "educations",
        "position",
        "fullPositions",
        "skills",
        "volunteering",
        "id",
        "country",
        "city",
        "countryCode",
    ]
    final_df.drop(
        columns=[col for col in cols_to_drop if col in final_df.columns], inplace=True
    )

    # --- Step 4: Save to People Collection ---
    if not final_df.empty:
        update_operations = []
        for record in final_df.to_dict("records"):
            record["message_date"] = datetime.utcnow()
            update_operations.append(
                UpdateOne(
                    {"profileURL": record["profileURL"]}, {"$set": record}, upsert=True
                )
            )

        if update_operations:
            db.People.bulk_write(update_operations)
        logger.info(f"TASK 3: Successfully processed and saved {len(final_df)} people.")
        people_urls = final_df["profileURL"].dropna().unique().tolist()
        return (people_urls, valid_job_ids)

    return ([], valid_job_ids)


@celery_app.task(
    bind=True,
    autoretry_for=(RapidAPIError,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
)
def fetch_posts_task(self, prev_result: tuple, pipeline_params: dict):
    """Task 4: Fetches recent posts for the processed people."""
    people_urls, valid_job_ids = prev_result
    run_id = pipeline_params.get("run_id")
    update_status(run_id, "Fetching Posts...")
    if not people_urls:
        return people_urls

    api_client = RapidAPIClient(api_key=RAPID_API_KEY)
    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]

    people_cursor = db.People.find(
        {"profileURL": {"$in": people_urls}}, {"username": 1}
    )
    people_usernames = [p["username"] for p in people_cursor if "username" in p]

    all_posts = []
    for username in people_usernames:
        # Check for existing posts first
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        existing_posts = list(
            db.RawPosts.find(
                {
                    "username": username,
                    "postedDateTimestamp": {"$gte": cutoff_date.timestamp()},
                },
                {"_id": 0},
            )
        )

        if existing_posts:
            logger.info(
                f"Found {len(existing_posts)} recent posts in DB for {username}"
            )
            all_posts.extend(existing_posts)
        else:
            logger.info(f"No recent posts in DB for {username}, calling API...")
            try:
                posts = api_client.get_linkedin_posts_by_username(username)
                if posts and posts.get("data"):
                    user_posts = posts["data"]
                    for post in user_posts:
                        post["username"] = username
                        post["message_date"] = datetime.utcnow()
                    all_posts.extend(user_posts)

                    # Upsert new posts into RawPosts based on postUrl
                    if user_posts:
                        operations = [
                            UpdateOne(
                                {"postUrl": post["postUrl"]},
                                {"$set": post},
                                upsert=True,
                            )
                            for post in user_posts
                            if "postUrl" in post
                        ]
                        if operations:
                            db.RawPosts.bulk_write(operations, ordered=False)

                time.sleep(1)
            except RapidAPIError as e:
                logger.error(f"Could not fetch posts for {username}: {e}")
                continue

    if not all_posts:
        logger.info("No posts found for any user.")
        return people_urls

    posts_df = pd.DataFrame(all_posts)

    # Clean posts
    drop_columns = [
        "isBrandPartnership",
        "totalReactionCount",
        "likeCount",
        "empathyCount",
        "commentsCount",
        "shareUrl",
        "postedAt",
        "urn",
        "company",
        "document",
        "celebration",
        "entity",
        "companyMentions",
        "InterestCount",
        "praiseCount",
        "repostsCount",
        "image",
        "mentions",
        "appreciationCount",
        "video",
        "funnyCount",
        "article",
    ]
    posts_df.drop(columns=drop_columns, inplace=True, errors="ignore")
    posts_df = posts_df.assign(
        repost=posts_df.get("reposted", False).fillna(False),
        author=posts_df["author"].apply(process_post_author),
        resharedPost=posts_df["resharedPost"].apply(process_reshared_data),
    ).rename(columns={"Username": "username"})

    # Upsert cleaned posts into the 'Posts' collection
    if not posts_df.empty:
        update_operations = []
        for record in posts_df.to_dict("records"):
            if "postUrl" in record:
                record["message_date"] = datetime.utcnow()
                update_operations.append(
                    UpdateOne(
                        {"postUrl": record["postUrl"]}, {"$set": record}, upsert=True
                    )
                )
        if update_operations:
            db.Posts.bulk_write(update_operations)

    logger.info(f"TASK 4: Successfully processed {len(posts_df)} posts.")
    return (people_urls, valid_job_ids)


@celery_app.task(
    bind=True,
    autoretry_for=(SnovError,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
)
def find_emails_task(self, prev_result: tuple, pipeline_params: dict):
    """Task 5: Finds emails for the given people using Snov.io, with enrichment."""
    people_urls, valid_job_ids = prev_result
    run_id = pipeline_params.get("run_id")
    update_status(run_id, "Finding Emails (Step 1/2)...")
    if not people_urls:
        return people_urls

    snov_client = SnovClient(client_id=SNOV_CLIENT_ID, client_secret=SNOV_CLIENT_SECRET)
    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]

    # --- Step 1: Find by Profile URL ---
    for profile_url in people_urls:
        try:
            person = db.People.find_one({"profileURL": profile_url})
            if person and person.get("snov_emails"):
                continue  # Skip if already has emails

            snov_data = db.RawSnovData.find_one({"profileURL": profile_url})
            if not snov_data:
                snov_data = snov_client.get_profile_by_url(profile_url)
                if snov_data and snov_data.get("success"):
                    snov_data["profileURL"] = profile_url
                    snov_data["message_date"] = datetime.utcnow()
                    db.RawSnovData.update_one(
                        {"profileURL": profile_url}, {"$set": snov_data}, upsert=True
                    )

            if snov_data and snov_data.get("success"):
                emails = [
                    e["email"]
                    for e in snov_data.get("data", {}).get("emails", [])
                    if "email" in e
                ]
                if emails:
                    db.People.update_one(
                        {"profileURL": profile_url},
                        {
                            "$set": {
                                "snov_emails": emails,
                                "snov_data_retrieved": True,
                                "message_date": datetime.utcnow(),
                            }
                        },
                    )
        except SnovError as e:
            logger.error(f"Snov API error (URL search) for {profile_url}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error in Step 1 for {profile_url}: {e}")
            continue

    # --- Step 2: Enrich Missing Emails using Domain Search ---
    update_status(run_id, "Enriching Missing Emails (Step 2/2)...")

    people_data = list(
        db.RawSnovData.find(
            {"profileURL": {"$in": people_urls}}, {"_id": 0, "success": 0}
        )
    )

    for record in people_data:
        if "profileURL" in record:
            record.setdefault("data", {})
            record["data"]["profileURL"] = record["profileURL"]
            del record["profileURL"]

    flattened_people = []
    for record in people_data:
        flat = flatten_person_data(record.get("data", {}))
        emails = extract_emails(flat.get("emails"))

        if not emails:
            first_name = flat.get("firstName")
            last_name = flat.get("lastName")
            domain = extract_domain(flat.get("currentJob_site"))

            if all([first_name, last_name, domain]):
                flattened_people.append(
                    {
                        "profileURL": flat.get("profileURL"),
                        "first_name": first_name,
                        "last_name": last_name,
                        "domain": domain,
                    }
                )

    for person in flattened_people:
        try:
            task_data = snov_client.find_emails_by_name_and_domain(
                first_name=person["first_name"],
                last_name=person["last_name"],
                domain=person["domain"],
            )
            task_hash = task_data.get("data", [{}])[0].get("task_hash")
            if not task_hash:
                continue

            time.sleep(30)  # Wait for async task to complete

            email_data = snov_client.get_email_finder_result(task_hash)
            if email_data.get("success") and email_data.get("data"):
                email_result = email_data["data"][0].get("result", [{}])[0]
                found_email = email_result.get("email")

                if found_email:
                    db.People.update_one(
                        {"profileURL": person["profileURL"]},
                        {
                            "$set": {
                                "snov_emails": [found_email],
                                "snov_data_retrieved": True,
                                "message_date": datetime.utcnow(),
                            }
                        },
                    )

                    # 🔽 Upsert domain enrichment data to RawSnovData
                    db.RawSnovData.update_one(
                        {"profileURL": person["profileURL"]},
                        {
                            "$set": {
                                "domain_search_result": email_data,
                                "domain_search_timestamp": datetime.utcnow(),
                                "message_date": datetime.utcnow(),
                            }
                        },
                        upsert=True,
                    )

        except SnovError as e:
            logger.error(
                f"Snov API error (Domain search) for {person['profileURL']}: {e}"
            )
            continue
        except Exception as e:
            logger.error(
                f"Unexpected error in domain enrichment for {person['profileURL']}: {e}"
            )
            continue

    return (people_urls, valid_job_ids)


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 120},
)
def generate_questionnaire_task(self, prev_result: tuple, pipeline_params: dict):
    """Task 6: Generates questionnaire data using Claude for each prospect."""
    people_urls, valid_job_ids = prev_result
    update_status(pipeline_params.get("run_id"), "Generating Questionnaires...")
    if not people_urls:
        return people_urls

    logger.info(f"TASK 6: Generating questionnaires for {len(people_urls)} people.")
    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]
    claude_client = ClaudeClient()

    people_data = list(db.People.find({"profileURL": {"$in": people_urls}}, {"_id": 0}))

    for person in people_data:
        try:
            username = person.get("username")
            company_name = person.get("companyName")
            if not company_name or not username:
                logger.warning(f"Missing company_name or username for person: {person}")
                continue

            job_data = (
                db.Jobs.find_one({"company_name": company_name}, {"_id": 0}) or {}
            )
            company_data = db.Company.find_one({"name": company_name}, {"_id": 0}) or {}

            posts_data = list(db.Posts.find({"username": username}, {"_id": 0})) or []

            people_df = pd.DataFrame([person])
            posts_df = pd.DataFrame(posts_data)

            if people_df.empty:
                logger.warning(f"People data empty for {username}")
                continue

            # Defensive clean_data_for_questionnaire
            try:
                people_dict, recent_posts = clean_data_for_questionnaire(
                    people_df, posts_df
                )
            except Exception as ce:
                logger.error(f"Error cleaning data for {username}: {ce}", exc_info=True)
                continue

            person_str = json.dumps(people_dict.get("Prospect 1", {}), default=str)
            posts_str = json.dumps(
                recent_posts.to_dict("records") if not recent_posts.empty else [],
                default=str,
            )
            job_str = json.dumps(job_data, default=str)
            company_str = json.dumps(company_data, default=str)

            system_prompt, user_prompt = get_question_summarizer_prompts(
                job_str, company_str, person_str, posts_str
            )

            logger.info(f"Calling Claude for {username} at {company_name}...")
            response_text, total_cost = claude_client.get_structured_response(
                system_prompt, user_prompt
            )

            qa_list = parse_llm_json_response(response_text)

            if qa_list:
                result_doc = {
                    "username": username,
                    "company_name": company_name,
                    "questionnaire": qa_list,
                    "llm_cost_usd": round(total_cost, 5),
                    "generated_at": pd.Timestamp.utcnow(),
                    "message_date": pd.Timestamp.utcnow(),
                }

                db.QuestionnaireResults.update_one(
                    {"username": username}, {"$set": result_doc}, upsert=True
                )
                logger.info(f"✅ Successfully upserted questionnaire for {username}.")
            else:
                logger.warning(
                    f"⚠️ Could not parse valid JSON from Claude for {username}"
                )

        except Exception as e:
            logger.error(
                f"❌ Failed to generate questionnaire for {person.get('username')}: {e}",
                exc_info=True,
            )

    logger.info("TASK 6: Finished questionnaire generation.")
    return (people_urls, valid_job_ids)


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 1, "countdown": 300},
)
def generate_email_sequence_task(self, prev_result: tuple, pipeline_params: dict):
    """
    Task 7: Runs the email generation agent for each prospect.
    """
    people_urls, valid_job_ids = prev_result
    update_status(pipeline_params.get("run_id"), "Generating Email Sequences...")
    if not people_urls:
        return people_urls

    logger.info(
        f"TASK 7: Starting email generation agent for {len(people_urls)} prospects."
    )

    async def main_async_loop():
        """Helper async function to run the agent calls in a single event loop."""
        for url in people_urls:
            try:
                logger.info(f"--- Running email agent for prospect: {url} ---")
                await generate_email_sequence(url)
                logger.info(f"--- Finished email agent for prospect: {url} ---")
            except Exception as e:
                # Log the error for the specific URL but continue with the next one
                logger.error(
                    f"Email generation agent failed for URL {url}: {e}", exc_info=True
                )
                continue

    # Run the main async function once. This creates one event loop for all URLs.
    try:
        asyncio.run(main_async_loop())
    except Exception as e:
        logger.error(
            f"A critical error occurred in the email generation task: {e}",
            exc_info=True,
        )

    logger.info("TASK 7: Finished running all email generation agents.")
    return (people_urls, valid_job_ids)


@celery_app.task(
    bind=True,
    autoretry_for=(SnovError, Exception),
    retry_kwargs={"max_retries": 2, "countdown": 60},
)
def dump_data_to_snov_task(self, prev_result: tuple, pipeline_params: dict):
    """Task 8 (Final): Gathers all data for each prospect and dumps it to Snov.io."""
    people_urls, valid_job_ids = prev_result
    run_id = pipeline_params.get("run_id")
    snov_list_id = pipeline_params.get("snov_list_id")
    update_status(run_id, "Dumping Data to Snov.io...")
    if not people_urls:
        update_status(run_id, "Completed: No prospects to dump.")
        return "Pipeline finished: No prospects to dump."

    snov_client = SnovClient(client_id=SNOV_CLIENT_ID, client_secret=SNOV_CLIENT_SECRET)
    db = MongoClient(MONGO_URI)[MONGO_DB_NAME]

    QUESTION_TO_SNOV_FIELD = {
        "What is the full legal name of the company?": "What is the full legal name of the company?",
        "What industry or niche do they primarily operate in?": "What industry or niche do they primarily operate in?",
        "Where is the company headquartered (city & country)?": "Where is the company headquartered (city & country)?",
        "What is the current estimated employee count?": "What is the current estimated employee count?",
        "What is the company website URL as mentioned in the LinkedIn Data?": "What is the company website URL as mentioned in the LinkedIn Data?",
        "Name of the individual": "Name of the individual",
        "Their job title or designation": "Their job title or designation",
        "Are they likely a decision-maker (e.g., manager, VP, director, CXO)?": "Are they likely a decision-maker (e.g., manager, VP, director, CXO)?",
        "Are they hiring for roles that suggest growth, scaling, or specific operational challenges? If yes, mention the roles.": "Are they hiring for roles that suggest growth, scaling, or specific operational challenges?",
        "Based on the company's industry, employee size, and growth stage, does it align with the Ideal Customer Profile (ICP) outlined in the ICP definition? Provide a brief rationale for your assessment.": "Based on the company's industry, employee size, and growth stage, does it align with the ICP?",
        "Are they likely to have the budget and maturity to engage with our service/product?": "Are they likely to have the budget and maturity to engage with our service/product?",
        "Have they posted or reshared any content that shows their pain points or areas of focus? Summarize relevant content if available.": "Have they posted or reshared any content that shows their pain points or areas of focus?",
        "Mention any known external tools or platforms the company uses (e.g., CRMs, marketing automation, cloud platforms, AI tools).": "Mention any known external tools or platforms the company uses.",
        "Can you derive a clear value proposition we might be able to offer, based on their context?": "Can you derive a clear value proposition we might be able to offer, based on their context?",
    }

    prospects = list(db.People.find({"profileURL": {"$in": people_urls}}))
    questionnaires = {
        q["username"]: q["questionnaire"]
        for q in db.QuestionnaireResults.find(
            {"username": {"$in": [p.get("username") for p in prospects]}}
        )
    }
    email_sequences = {
        e["linkedin_url"]: e
        for e in db.GeneratedEmails.find({"linkedin_url": {"$in": people_urls}})
    }
    jobs_map = {
        job["company_name"]: job
        for job in db.Jobs.find({"job_id": {"$in": valid_job_ids}})
    }

    for prospect in prospects:
        try:
            profile_url = prospect.get("profileURL")
            if not profile_url:
                continue

            prospect_company = prospect.get("companyName")
            relevant_job = jobs_map.get(prospect_company, {})

            payload = {
                "email": prospect["snov_emails"][0]
                if prospect.get("snov_emails")
                else f"{prospect.get('firstName', 'user').strip().lower().replace(' ', '')}@yopmail.com",
                "firstName": prospect.get("firstName"),
                "lastName": prospect.get("lastName"),
                "fullName": f"{prospect.get('firstName', '')} {prospect.get('lastName', '')}".strip(),
                "country": prospect["location"].strip().split()[-1]
                if prospect.get("location")
                else None,
                "location": prospect["location"],
                "position": prospect.get("title"),
                "companyName": prospect.get("companyName"),
                "companySite": prospect.get("companyURL"),
                "socialLinks[linkedIn]": profile_url,
                "customFields[LinkedIn_Job_URL]": relevant_job.get("job_url"),
                "customFields[Open_Role]": relevant_job.get("title"),
            }
            prospect_questionnaire = questionnaires.get(prospect.get("username"), [])
            for qa_pair in prospect_questionnaire:
                question = qa_pair.get("question")
                answer = qa_pair.get("answer")
                snov_field = QUESTION_TO_SNOV_FIELD.get(question)
                if snov_field:
                    payload[f"customFields[{snov_field}]"] = answer

            prospect_emails = email_sequences.get(profile_url, {})
            for i in range(1, 6):
                payload[f"customFields[Subject{i}]"] = prospect_emails.get(
                    f"subject_{i}"
                )
                payload[f"customFields[Email{i}]"] = prospect_emails.get(f"body_{i}")

            final_payload = {k: v for k, v in payload.items() if v is not None}

            # --- Save to FinalData collection in MongoDB ---
            mongo_payload = final_payload.copy()
            mongo_payload["listId"] = snov_list_id
            mongo_payload["message_date"] = datetime.utcnow()
            db.FinalData.update_one(
                {"profileURL": profile_url},
                {"$set": mongo_payload},
                upsert=True,
            )

            response = snov_client.add_prospect_to_list(snov_list_id, final_payload)

            if response.get("success"):
                logger.info(
                    f"✅ Success: {payload['fullName']} ({payload['email']}) added to Snov."
                )
            else:
                logger.error(
                    f"❌ Failed: {payload['fullName']} - Reason: {response.get('message')}"
                )
        except Exception as e:
            logger.error(
                f"Error processing prospect {prospect.get('username')} for Snov dump: {e}",
                exc_info=True,
            )
            continue

    update_status(run_id, "Completed Successfully")
    return "Pipeline finished successfully."


@celery_app.task
def start_lead_generation_pipeline(pipeline_params: dict):
    """
    This is the main entry point task. It kicks off the first loop.
    """
    run_id = pipeline_params.get("run_id")
    logger.info(f"--- LEAD GENERATION PIPELINE TRIGGERED (RUN ID: {run_id}) ---")
    update_status(run_id, "Pipeline Started")
    start_lead_generation_loop.delay(pipeline_params=pipeline_params)

    return {"status": "Pipeline loop initiated successfully!"}
