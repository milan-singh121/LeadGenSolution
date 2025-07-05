"""
This script contains the prompt-generation logic for the email generation agent.
It builds the prompts dynamically using data fetched from MongoDB.
"""

import json


def get_email_system_prompt(base_system_prompt: str, first_name: str) -> str:
    """
    Formats the base system prompt with the prospect's first name and adds
    strict HTML formatting instructions.
    """
    formatted_prompt = base_system_prompt.format(name=first_name)
    formatting_rules = """
    
    IMPORTANT FORMATTING RULES:
    1.  You MUST use <br><br> for all paragraph breaks in the email body.
    2.  To emphasize key information, such as statistics, results, or important phrases, you MUST wrap them in <b></b> tags. For example: "achieved a <b>40% increase</b> in user engagement".
    3.  Do NOT use markdown newlines (e.g., \\n) for formatting. The output must be clean HTML.
    4.  Ensure the final output is wrapped in the specified XML tags.
    5. Important Note: Ensure the email formatting is clean and professional. Use HTML line breaks (<br>) it's the most important necessary. Maintain proper spacing throughout the email, especially around bullet points. Apply single or double line breaks thoughtfully to enhance readability and structure.
    """
    return formatted_prompt + formatting_rules


def get_email_user_prompt(
    first_name: str,
    last_name: str,
    headline: str,
    summary: str,
    skills: list,
    company_industry: str,
    title: str,
    description: str,
    full_positions: list,
    recent_posts: list,
    job_data: dict,
    company_info: dict,
    client_data: list,
    email_blueprint: dict,
) -> str:
    """
    Constructs the detailed user prompt by combining all prospect, company,
    and blueprint data into a single, comprehensive context for the LLM.
    """
    # Convert complex data to JSON strings for clean insertion into the prompt
    job_data_str = json.dumps(job_data, indent=2, default=str)
    full_positions_str = json.dumps(full_positions, indent=2, default=str)
    recent_posts_str = json.dumps(recent_posts, indent=2, default=str)
    company_info_str = json.dumps(company_info, indent=2, default=str)
    client_data_str = json.dumps(client_data, indent=2, default=str)
    blueprint_str = json.dumps(email_blueprint, indent=2, default=str)

    # This prompt structure combines all data points into one context for the LLM
    return f"""
        The emails should always have html line breakers (<b></b>) present in them for proper formatting. Never forget to include them

        Here is the complete context for generating an email for {first_name}.

        **A. The Email Blueprint (Your Instructions for this specific email):**
        {blueprint_str}
        
        **B. Prospect's Core Information:**
        - Name: {first_name} {last_name}
        - Headline: {headline}
        - Current Role: {title} at a company in the **{company_industry}** industry.
        - LinkedIn Summary: {summary}
        - Key Skills: {skills}

        **C. Prospect's Professional Experience:**
        - Detailed Positions: {full_positions_str}
        
        **D. Prospect's Recent Activity (Their own words):**
        - Recent Posts/Reshares: {recent_posts_str}

        **E. Company Context & Potential Needs:**
        - Open Roles at Their Company: {job_data_str}
        - Job Description Snippet (for one of the roles): {description}

        **F. Our Company (The Sender - InHousen):**
        - About Us: {company_info_str}

        **G. Our Past Client Success Stories (for reference):**
        - Client Data: {client_data_str}

        Generate the email now based on ALL of the context above, following all formatting and content rules from the system prompt.

        Important Note: Ensure the email formatting is clean and professional. Use HTML line breaks (<br>) only where necessary. Maintain proper spacing throughout the email, especially around bullet points. Apply single or double line breaks thoughtfully to enhance readability and structure.
    """
