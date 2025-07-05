"""
This script defines the LangGraph state, nodes, and conditional edges
for the email generation agent, using Claude on Bedrock and robust XML parsing.
"""

import os
import sys
import logging
from typing import TypedDict, List, Optional
import json
import asyncio
import re
from langgraph.graph import StateGraph, END


app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Import tools, prompts, and the LLM client using relative imports
from email_agent.tools import (
    get_prospect_info,
    get_prospect_posts,
    get_company_info,
    get_email_blueprint,
    create_text_embedding,
    find_similar_clients,
)
from email_agent.prompt import get_email_system_prompt, get_email_user_prompt
from email_agent.agent_utils import extract_emails_from_llm_response
from email_agent.db_client import db as async_db  # Import the async db client
from clients.llm_clients import ClaudeClient

logger = logging.getLogger(__name__)
claude_client = ClaudeClient()


# --- State Definition ---
class EmailGenState(TypedDict):
    """Represents the state of our email generation graph."""

    prospect_url: str
    sequence_id: int
    prospect_name: Optional[str]
    prospect_info: Optional[dict]
    prospect_posts: Optional[List[dict]]
    company_info: Optional[dict]
    client_data: Optional[List[dict]]
    blueprint: Optional[dict]
    system_prompt: str
    draft_subject: Optional[str]
    draft_email: Optional[str]
    qa_feedback: Optional[str]
    rewrite_count: int
    total_cost: float


# --- Graph Nodes ---


async def research_node(state: EmailGenState) -> dict:
    """Gathers all necessary data from DB to build the prompts."""
    print(f"\n---NODE: Researching Prospect for Sequence #{state['sequence_id']}---")
    prospect_url = state["prospect_url"]
    sequence_id = state["sequence_id"]

    # Fetch all data concurrently using the async client
    prospect_info_task = get_prospect_info.ainvoke({"linkedin_url": prospect_url})
    company_info_task = get_company_info.ainvoke({})
    blueprint_task = get_email_blueprint.ainvoke({"sequence_id": sequence_id})
    system_prompt_doc_task = async_db.PromptInstructions.find_one(
        {"name": "email_system_prompt"}
    )

    prospect_info, company_info, blueprint, system_prompt_doc = await asyncio.gather(
        prospect_info_task, company_info_task, blueprint_task, system_prompt_doc_task
    )

    if prospect_info.get("error") or not system_prompt_doc:
        logger.error("Research node failed: Prospect or System Prompt not found.")
        return {"prospect_info": prospect_info, "system_prompt": None}

    full_name = f"{prospect_info.get('firstName', '')} {prospect_info.get('lastName', '')}".strip()

    # Vector search for similar clients
    similar_clients = []
    if prospect_industry := prospect_info.get("companyIndustry"):
        if industry_embedding := create_text_embedding.invoke(
            {"text": prospect_industry}
        ):
            similar_clients = await find_similar_clients.ainvoke(
                {"embedding": industry_embedding}
            )

    return {
        "prospect_info": prospect_info,
        "prospect_name": full_name or "there",
        "prospect_posts": await get_prospect_posts.ainvoke(
            {"username": prospect_info.get("username")}
        )
        if prospect_info.get("username")
        else [],
        "company_info": company_info,
        "blueprint": blueprint,
        "client_data": similar_clients,
        "system_prompt": system_prompt_doc["prompt_text"],
        "total_cost": 0.0,
        "rewrite_count": 0,
    }


async def draft_email_node(state: EmailGenState) -> dict:
    """Drafts the initial email using the database-driven prompts and XML parsing."""
    print("---NODE: Drafting Email---")
    if not all(state.get(k) for k in ["prospect_name", "blueprint", "system_prompt"]):
        return {
            "draft_email": "Error: Missing critical data for drafting.",
            "draft_subject": "Error",
        }

    person = state["prospect_info"]
    company = person.get("company")
    job_data = (
        await async_db.Jobs.find_one({"company_name": company}) if company else {}
    )

    user_prompt = get_email_user_prompt(
        first_name=person.get("firstName", ""),
        last_name=person.get("lastName", ""),
        headline=person.get("headline", ""),
        summary=person.get("summary", ""),
        skills=person.get("clean_skills", []),
        company_industry=person.get("companyIndustry", ""),
        title=person.get("title", ""),
        description=person.get("description", ""),
        full_positions=person.get("processed_fullPositions", []),
        recent_posts=state["prospect_posts"],
        job_data=job_data,
        company_info=state["company_info"],
        client_data=state["client_data"],
        email_blueprint=state["blueprint"],
    )
    system_prompt = get_email_system_prompt(
        state["system_prompt"], state["prospect_name"]
    )

    response_text, cost = claude_client.get_structured_response(
        system_prompt, user_prompt
    )
    emails = extract_emails_from_llm_response(response_text)

    if emails:
        draft = emails[0]
        return {
            "draft_subject": draft["subject"],
            "draft_email": draft["body"],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "total_cost": state.get("total_cost", 0) + cost,
        }
    logger.error("Failed to parse Claude's draft response into XML.")
    return {"draft_subject": "Error", "draft_email": "Parsing Error"}


# async def qa_node(state: EmailGenState) -> dict:
#     """Performs quality assurance on the drafted email."""
#     print("---NODE: Quality Assurance---")
#     logger.info(f"Email :, {state['draft_email']}")
#     qa_user_prompt = f"""
#         Please review the following email draft against the provided blueprint.

#         **Blueprint:**
#         {json.dumps(state["blueprint"], indent=2)}

#         **Draft Subject:** {state["draft_subject"]}
#         **Draft Body:**
#         {state["draft_email"]}


#         - If the email meets all the criteria, respond only with the word “APPROVED”. Otherwise, provide a numbered list of specific changes needed.
#         - When referencing the prospect’s personal or company details (such as name, job title, company name, or industry), always ensure they are accurate and match the provided prospect data exactly. Double-check for correct spelling, formatting, and context relevance to maintain a personalized and professional tone throughout the email.
#     """
#     feedback, cost = claude_client.get_structured_response(
#         state["system_prompt"], qa_user_prompt
#     )
#     print(f"---QA Feedback: {feedback}---")
#     return {"qa_feedback": feedback, "total_cost": state.get("total_cost", 0) + cost}


async def qa_node(state: EmailGenState) -> dict:
    """Performs quality assurance on the drafted email with updated prompt."""
    print("---NODE: Quality Assurance---")
    logger.info(f"QA review for email: {state['draft_email']}")
    qa_user_prompt = f"""
        Please review the following email draft against the provided blueprint.
        
        **Blueprint:**
        {json.dumps(state["blueprint"], indent=2)}
        
        **Draft Subject:** {state["draft_subject"]}
        **Draft Body:**
        {state["draft_email"]}

        **Instructions:**
        1. First, provide a numbered list of specific, actionable changes needed to improve the email.
        2. After the list, provide a final summary sentence that MUST start with "Overall, the email meets XX% of the criteria." where XX is your estimated percentage of how well the draft adheres to the blueprint.
        3. If and only if the email is 100% perfect and needs no changes, respond ONLY with the word "APPROVED".
        4. When referencing the prospect’s personal or company details (such as name, job title, company name, or industry), always ensure they are accurate and match the provided prospect data exactly. Double-check for correct spelling, formatting, and context relevance to maintain a personalized and professional tone throughout the email.
    """
    feedback, cost = claude_client.get_structured_response(
        state["system_prompt"], qa_user_prompt
    )
    print(f"---QA Feedback: {feedback}---")
    return {"qa_feedback": feedback, "total_cost": state.get("total_cost", 0) + cost}


async def rewrite_node(state: EmailGenState) -> dict:
    """Rewrites the email based on QA feedback."""
    print("---NODE: Rewriting Email---")
    rewrite_user_prompt = f"""
        Please rewrite the following email based on the QA feedback.
        
        **QA Feedback:**
        {state["qa_feedback"]}
        
        **Original Draft Subject:** {state["draft_subject"]}
        **Original Draft Body:**
        {state["draft_email"]}

        **Original Blueprint:**
        {json.dumps(state["blueprint"], indent=2)}
        
        Provide only the rewritten email in the required XML format, inside <emails> and <email> tags.
    """
    response_text, cost = claude_client.get_structured_response(
        state["system_prompt"], rewrite_user_prompt
    )
    emails = extract_emails_from_llm_response(response_text)

    if emails:
        draft = emails[0]
        return {
            "draft_subject": draft["subject"],
            "draft_email": draft["body"],
            "rewrite_count": state["rewrite_count"] + 1,
            "total_cost": state.get("total_cost", 0) + cost,
        }
    logger.error("Failed to parse Claude's rewrite response into XML.")
    return {"draft_subject": "Error", "draft_email": "Rewrite Parsing Error"}


# --- Conditional Edges & Graph Definition ---
def prospect_found(state: EmailGenState) -> str:
    if "error" in state.get("prospect_info", {}) or not state.get("system_prompt"):
        return "end"
    return "continue"


# def should_rewrite(state: EmailGenState) -> str:
#     if "APPROVED" in state["qa_feedback"] or state["rewrite_count"] >= 3:
#         return "end"
#     return "rewrite"


# def should_rewrite(state: EmailGenState) -> str:
#     """
#     Decision node to determine the next step after QA based on a percentage score.
#     """
#     print("---NODE: Making Decision on Rewrite---")
#     feedback = state["qa_feedback"]
#     rewrite_count = state["rewrite_count"]

#     # Condition 1: Explicit approval (100% perfect)
#     if "APPROVED" in feedback:
#         print("---DECISION: Email explicitly APPROVED! Finishing flow.---")
#         return "end"

#     # Condition 2: Check for percentage score
#     match = re.search(r"meets (\d+)% of the criteria", feedback, re.IGNORECASE)
#     if match:
#         score = int(match.group(1))
#         print(f"---DECISION: QA Agent scored the email at {score}%.---")
#         if score >= 75:
#             print("---DECISION: Score is >= 85%. Finishing flow.---")
#             return "end"

#         # # Condition 3: Max rewrites reached
#         # if rewrite_count >= 4:
#         #     print(
#         #         f"---DECISION: Max rewrites ({rewrite_count}) reached. Finishing flow.---"
#         #     )
#         # return "end"

#     # Otherwise, continue rewriting
#     print("---DECISION: Revisions needed. Rerouting to rewrite.---")
#     return "rewrite"


def should_rewrite(state: EmailGenState) -> str:
    """
    Decision node to determine the next step after QA based on a percentage score.
    The loop ends only if the score is >= 80%.
    """
    print("---NODE: Making Decision on Rewrite---")
    feedback = state["qa_feedback"]

    # Primary Condition: Check for percentage score
    match = re.search(r"meets (\d+)% of the criteria", feedback, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        print(f"---DECISION: QA Agent scored the email at {score}%.---")
        if score >= 80:
            print("---DECISION: Score is >= 80%. Finishing flow.---")
            return "end"

    # If score is < 80 or no score was found, continue rewriting indefinitely.
    print(
        "---DECISION: Score is < 80% or no score found. Revisions needed. Rerouting to rewrite.---"
    )
    return "rewrite"


def build_graph() -> StateGraph:
    workflow = StateGraph(EmailGenState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("draft_email", draft_email_node)
    workflow.add_node("qa_email", qa_node)
    workflow.add_node("rewrite_email", rewrite_node)
    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher", prospect_found, {"continue": "draft_email", "end": END}
    )
    workflow.add_edge("draft_email", "qa_email")
    workflow.add_conditional_edges(
        "qa_email", should_rewrite, {"rewrite": "rewrite_email", "end": END}
    )
    workflow.add_edge("rewrite_email", "qa_email")
    return workflow.compile()
