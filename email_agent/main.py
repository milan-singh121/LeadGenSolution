"""
This script contains the main execution logic for running the email
generation agent for a single prospect.
"""

import os
import sys
import asyncio
import logging
import pandas as pd

app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use relative imports for components within the same 'agents' package
from email_agent.graph_setup import build_graph, EmailGenState
from email_agent.tools import save_email_sequence

logger = logging.getLogger(__name__)


async def run_agent_for_sequence_step(prospect_url: str, sequence_id: int):
    """
    Runs the email personalization agent for a single sequence step.
    Returns the final state.
    """
    app = build_graph()
    initial_state = {
        "prospect_url": prospect_url,
        "sequence_id": sequence_id,
    }
    final_state: EmailGenState = {}

    # Stream the events from the graph execution
    async for event in app.astream(initial_state):
        for key, value in event.items():
            if key != "__end__":
                final_state.update(value)

    return final_state


async def generate_email_sequence(prospect_url: str):
    """
    Runs the email generation agent for a full sequence, accumulates the results,
    and saves them as a single document in MongoDB.
    """
    total_cost = 0.0
    generated_emails = {}

    # Loop through the desired number of emails in the sequence
    for i in range(1, 6):  # Generates 5 emails
        sequence_id = i
        final_state = await run_agent_for_sequence_step(prospect_url, sequence_id)

        # Accumulate cost from the state
        total_cost += final_state.get("total_cost", 0)

        final_subject = final_state.get("draft_subject")
        final_body = final_state.get("draft_email")

        if final_subject and final_body and "Error" not in final_subject:
            print(f"✅ Email #{sequence_id} generated successfully.")
            generated_emails[f"subject_{sequence_id}"] = final_subject
            generated_emails[f"body_{sequence_id}"] = final_body
        else:
            print(f"❌ Email #{sequence_id} generation failed. It will not be saved.")

    if generated_emails:
        print("\n--- Saving full email sequence to the database... ---")
        save_result = await save_email_sequence.ainvoke(
            {
                "linkedin_url": prospect_url,
                "emails": generated_emails,
                "message_date": pd.Timestamp.utcnow(),
                "total_cost": total_cost,
            }
        )
        print(f"--- Database save result: {save_result} ---")

    print(f"\n\nSEQUENCE COMPLETE. TOTAL ESTIMATED COST: ${total_cost:.5f}")
