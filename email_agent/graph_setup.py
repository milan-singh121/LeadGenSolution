"""
This script defines the LangGraph state, nodes, and conditional edges
for the email generation agent, using a Critic -> Reflect -> Refine loop.
"""

import os
import sys
import logging
import json
import asyncio
import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# Add project root to path
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
from email_agent.db_client import db as async_db
from clients.llm_clients import ClaudeClient  # Assuming you have this client

logger = logging.getLogger(__name__)
claude_client = ClaudeClient()


class EmailGenState(TypedDict):
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
    self_correction_plan: Optional[str]
    critic_feedback: Optional[str]
    refine_count: int
    total_cost: float


async def research_node(state: EmailGenState) -> dict:
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
        logger.error("Research failed: Prospect or System Prompt not found.")
        return {"prospect_info": prospect_info, "system_prompt": None}

    full_name = f"{prospect_info.get('firstName', '')} {prospect_info.get('lastName', '')}".strip()
    similar_clients = []
    if industry := prospect_info.get("companyIndustry"):
        if embedding := create_text_embedding.invoke({"text": industry}):
            similar_clients = await find_similar_clients.ainvoke(
                {"embedding": embedding}
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
        "refine_count": 0,
    }


async def draft_email_node(state: EmailGenState) -> dict:
    print("---NODE: Drafting Email (V1)---")
    if not all(state.get(k) for k in ["prospect_name", "blueprint", "system_prompt"]):
        return {"draft_email": "Error: Missing data", "draft_subject": "Error"}

    person = state["prospect_info"]
    company = person.get("company")
    job_data = (
        await async_db.jobs_collection.find_one({"company_name": company})
        if company
        else {}
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
            "refine_count": state.get("refine_count", 0) + 1,
            "total_cost": state.get("total_cost", 0) + cost,
        }

    logger.error("Failed to parse Claude's draft response into XML.")
    return {"draft_subject": "Error", "draft_email": "Parsing Error"}


async def self_correction_node(state: EmailGenState) -> dict:
    """Proactively reviews its own draft to create an improvement plan."""
    print("---NODE: Self-Correction Review---")
    self_correction_prompt = f"""
        You are a senior copy editor. You have just written the following draft.
        Review your own work critically against the provided blueprint and all source data.
        Your goal is to identify areas for improvement before it goes to a final check.

        **Non-Negotiable Rules:**
            1.  The email must NOT contain any placeholders like [Company Name], [First Name], [Signature], [Your Name], etc.
            2.  The email must NOT contain any data (names, statistics, facts) that cannot be verified from the 'Source Data' below.
            3.  The email must NOT mention client names or industry that are not present in the 'Client Data' section of the source data.
            4.  The email must NOT include a signature or sign-off like "Best," "Regards," or "P.S.".
            5.  The email must NOT make assumptions. All personalization must be tied directly to the provided source data.
            6.  The prospect's first name, "{state["prospect_name"]}", must be used correctly. It is acceptable to use the name more than once if it enhances personalization.
            7.  Word count is flexible; do not fail the email based on its length as long as it is coherent and professional.
            8.  The Email must NOT mention "AI-powered solutions" as the content of the email, it should be more personlized for the prospects need.
            9.	The email must not mention “AI-powered solutions” or imply that we have developed any tools. We are a staffing company that provides qualified candidates, not a tech provider or product company.

        **Source Data:**
            - Prospect Info: {json.dumps(state["prospect_info"], indent=2)}
            - Prospect's LinkedIn Posts: {json.dumps(state["prospect_posts"], indent=2)}
            - Client Data: {json.dumps(state["client_data"], indent=2)}
            - Blueprint: {json.dumps(state["blueprint"], indent=2)}
            - About Inhousen: {json.dumps(state["company_info"], indent=2)}
            - System Prompt: {json.dumps(state["system_prompt"], indent=2)}
        --- END SOURCE DATA ---
        
        **Your Draft to Review:**
            Subject: {state["draft_subject"]}
            Body: {state["draft_email"]}

        Create a concise, bulleted list of actionable instructions for a junior copywriter to refine this draft into a final, high-quality version.
        This plan should address any rule violations and also suggest improvements to tone, flow, and persuasiveness.
        If the draft is already perfect, simply respond with "No changes needed."
    """
    plan, cost = claude_client.get_structured_response(
        state["system_prompt"], self_correction_prompt
    )
    print(f"---Self-Correction Plan:\n{plan}---")
    return {
        "self_correction_plan": plan,
        "total_cost": state.get("total_cost", 0) + cost,
    }


async def refine_node(state: EmailGenState) -> dict:
    """Refines the email based on the self-correction plan."""
    print("---NODE: Refining Email (V2)---")
    refine_user_prompt = f"""
        Please rewrite the following email. You must strictly follow the 'Self-Correction Plan' to create a superior version.

        **Self-Correction Plan (Your guide):**
        {state["self_correction_plan"]}
        
        **Original Draft Subject:** {state["draft_subject"]}
        **Original Draft Body:**
        {state["draft_email"]}

        **Original Blueprint (for reference):**
        {json.dumps(state["blueprint"], indent=2)}
        
        Provide only the rewritten email in the required XML format, inside <emails> and <email> tags.
    """
    response_text, cost = claude_client.get_structured_response(
        state["system_prompt"], refine_user_prompt
    )
    emails = extract_emails_from_llm_response(response_text)
    if emails:
        draft = emails[0]
        return {
            "draft_subject": draft["subject"],
            "draft_email": draft["body"],
            "refine_count": state["refine_count"] + 1,
            "total_cost": state.get("total_cost", 0) + cost,
        }
    logger.error("Failed to parse Claude's refine response into XML.")
    return {"draft_subject": "Error", "draft_email": "Refine Parsing Error"}


async def correction_review_node(state: EmailGenState) -> dict:
    """
    NEW NODE: Checks if the refined email successfully implemented the self-correction plan.
    """
    print("---NODE: Correction Review---")
    review_prompt = f"""
        You are a meticulous reviewer. Your only job is to check if the 'Refined Email' has successfully implemented all the instructions from the 'Self-Correction Plan'.

        **Self-Correction Plan (The instructions that MUST be followed):**
        {state["self_correction_plan"]}

        **Refined Email to Review:**
        Subject: {state["draft_subject"]}
        Body: {state["draft_email"]}

        Compare the refined email to the plan. If all points in the plan have been addressed, respond ONLY with the word "PASS".
        If any instruction from the plan was not followed, respond with "FAIL" and a one-sentence explanation of which instruction was missed.
    """
    feedback, cost = claude_client.get_structured_response(
        state["system_prompt"], review_prompt
    )
    print(f"---Correction Review Feedback: {feedback}---")
    return {
        "correction_review_feedback": feedback,
        "total_cost": state.get("total_cost", 0) + cost,
    }


async def critic_node(state: EmailGenState) -> dict:
    """Acts as the final gatekeeper, checking the *refined* draft against non-negotiable rules."""
    print("---NODE: Final Critic Review---")
    critic_prompt = f"""
        You are a final proofreader. Your only job is to give a final check on the following email against the non-negotiable rules.
        Use the provided source data for verification. If the email violates ANY rule, respond with "FAIL" and a one-sentence explanation.
        If the email is perfect and follows all rules, respond ONLY with the word "PASS".

        **Non-Negotiable Rules:**
            1.  The email must NOT contain any placeholders like [Company Name], [First Name], [Signature], [Your Name], etc.
            2.  The email must NOT contain any data (names, statistics, facts) that cannot be verified from the 'Source Data' below.
            3.  The email must NOT mention client names or industry that are not present in the 'Client Data' section of the source data.
            4.  The email must NOT include a signature or sign-off like "Best," "Regards," or "P.S.".
            5.  The email must NOT make assumptions. All personalization must be tied directly to the provided source data.
            6.  The prospect's first name, "{state["prospect_name"]}", must be used correctly. It is acceptable to use the name more than once if it enhances personalization.
            7.  Word count is flexible; do not fail the email based on its length as long as it is coherent and professional.
            8.  The Email must NOT mention "AI-powered solutions" as the content of the email, it should be more personlized for the prospects need.
            9.	The email must not mention “AI-powered solutions” or imply that we have developed any tools. We are a staffing company that provides qualified candidates, not a tech provider or product company.

        **Source Data:**
            - Prospect Info: {json.dumps(state["prospect_info"], indent=2)}
            - Prospect's LinkedIn Posts: {json.dumps(state["prospect_posts"], indent=2)}
            - Client Data: {json.dumps(state["client_data"], indent=2)}
            - Blueprint: {json.dumps(state["blueprint"], indent=2)}
            - About Inhousen: {json.dumps(state["company_info"], indent=2)}
            - System Prompt: {json.dumps(state["system_prompt"], indent=2)}
        --- END SOURCE DATA ---

        **Email Draft to Review:**
        Subject: {state["draft_subject"]}
        Body: {state["draft_email"]}
    """
    feedback, cost = claude_client.get_structured_response(
        state["system_prompt"], critic_prompt
    )
    print(f"---Critic Feedback: {feedback}---")
    return {
        "critic_feedback": feedback,
        "total_cost": state.get("total_cost", 0) + cost,
    }


def prospect_found(state: EmailGenState) -> str:
    return (
        "end"
        if "error" in state.get("prospect_info", {}) or not state.get("system_prompt")
        else "continue"
    )


def did_correction_pass(state: EmailGenState) -> str:
    """Checks if the self-correction plan was successfully implemented."""
    print("---NODE: Making Decision on Correction Review---")
    refine_count = state["refine_count"]

    if "PASS" in state.get("correction_review_feedback", ""):
        print("---DECISION: Correction plan PASSED. Proceeding to final critic.---")
        return "pass"

    if refine_count >= 4:  # 1 draft + 2 refinement attempts
        print(
            f"---DECISION: Max refinements ({refine_count}) reached. Finishing flow despite correction FAIL.---"
        )
        return "fail"  # End the loop even if it fails

    print("---DECISION: Correction plan FAILED. Rerouting for another refinement.---")
    return "refine"


def did_critic_pass(state: EmailGenState) -> str:
    """A final check. This should almost always pass."""
    if "PASS" in state.get("critic_feedback", ""):
        print("---DECISION: Final Critic PASSED. Finishing flow.---")
        return "end"
    else:
        print("---DECISION: Final Critic FAILED. Finishing flow with failed draft.---")
        return "end"


def build_graph() -> StateGraph:
    workflow = StateGraph(EmailGenState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("draft_email", draft_email_node)
    workflow.add_node("self_correction", self_correction_node)
    workflow.add_node("refine_email", refine_node)
    workflow.add_node("correction_review", correction_review_node)  # New node
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher", prospect_found, {"continue": "draft_email", "end": END}
    )

    # New workflow: Draft -> Self-Correct -> Refine -> Correction Review -> (loop or Critic)
    workflow.add_edge("draft_email", "self_correction")
    workflow.add_edge("self_correction", "refine_email")
    workflow.add_edge("refine_email", "correction_review")

    workflow.add_conditional_edges(
        "correction_review",
        did_correction_pass,
        {
            "pass": "critic",  # If correction passes, go to final critic
            "refine": "refine_email",  # If it fails, loop back to refine
            "fail": "critic",  # If max retries hit, go to critic anyway
        },
    )

    workflow.add_conditional_edges(
        "critic",
        did_critic_pass,
        {
            "end": END,
            "refine": "refine_email",  # Fallback loop, should rarely be used
        },
    )

    return workflow.compile()
