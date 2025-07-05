"""
This script contains all the prompts required for the Lead Gen Use Case.
The prompts are defined as static functions for easy import and use.
"""

import json


def get_icp_definition_prompt() -> str:
    """
    Returns the Ideal Customer Profile (ICP) definition prompt text.
    """
    return """
        Define Ideal Customer Profile (ICP) Match
        
        You are a lead intelligence assistant. Based on the following ICP definitions, assess whether a given company is a good fit.
        Evaluate based on three core criteria: Industry & Sector, Company Size, and Company Stage.
        
        ⸻
        
        ICP Definitions
        
        1. Industry & Sector
        The company should operate in one or more of the following industries:
            • Fintech (neo-banks, payment processors, online investment platforms, RegTech)
            • Complex B2B SaaS (e.g., cybersecurity platforms, CRM/ERP solutions, niche industry software)
            • Enterprise Software
            • E-commerce (online retail, marketplaces)
            • Digital Retail (traditional retail with strong online presence)
            • Manufacturing / Industrial Automation / Automotive / Heavy Machinery
            • Life Sciences (pharma, biotech, medical devices)
            • Digital Health / Healthcare SaaS / HealthTech SaaS (telehealth, diagnostics, remote monitoring, patient management)
            • Public Sector / Government Agencies / Public Utilities
        
        2. Company Size Categories
            • Category 1: 0–500 employees – Fintech Scale-ups, SaaS Startups, E-commerce
            • Category 2: 200–1000+ employees – Industrial/Manufacturing Enterprises
            • Category 3: 100–750 employees – Life Sciences, Digital Health companies
            • Category 4: 10–50 employees – Early-stage HealthTech SaaS startups
            • Category 5: 500–5000+ employees – Large public sector and government organizations
        
        3. Company Stage
            • Stage 1: Fintech Scale-up – Series A/B/C, growing 20%+ YoY
            • Stage 2: SaaS / Enterprise Scale-up – Mature scale-ups with enterprise clients
            • Stage 3: E-commerce/Digital Retail – Established players with seasonal cycles and digital focus
            • Stage 4: Industrial/Manufacturing – Mature companies driving digital transformation (e.g., Industry 4.0)
            • Stage 5: Life Sciences / Digital Health – R&D-heavy companies in new therapies/platforms
            • Stage 6: HealthTech SaaS Startup – Pre-Seed to Series A, product validation focus
            • Stage 7: Government/Public Innovation Units – Modernizing or piloting new tech with bureaucratic constraints
        """


def get_question_summarizer_prompts(
    job_data: str, company_data: str, prospect_data: str, prospects_posts_data: str
) -> (str, str):
    """
    Generates the system and user prompts for the question summarizer LLM.

    Args:
        job_data (str): String-formatted data about the job.
        company_data (str): String-formatted data about the company.
        prospect_data (str): String-formatted data about the prospect.
        prospects_posts_data (str): String-formatted data about the prospect's posts.

    Returns:
        A tuple containing the system prompt and the user prompt.
    """
    icp_definition = get_icp_definition_prompt()

    system_prompt = """
    You are a highly skilled B2B Sales Intelligence Assistant that analyzes semi-structured data about companies and people.
    Your job is to extract useful business insights and answer specific questions to help a B2B marketing and sales team qualify a lead.
    You must use only the data provided to answer each question. If the data is missing or unclear for a specific question, respond with: "Insufficient data to answer."
    You must always return your answer as a valid JSON list of question-answer objects. Do not return any question in the output where the data is insufficient to answer.
    Do not hallucinate.
    """

    user_prompt = f"""
    Given the following data, please answer the required questions.

    # ICP Definition:
    {icp_definition}

    # Provided Data:
    - Job Data: {job_data}
    - Company LinkedIn Profile: {company_data}
    - Prospect’s Professional Profile: {prospect_data}
    - Prospect’s Recent LinkedIn Activity (Posts/Reshares): {prospects_posts_data}

    # Questions to Answer:
    - What is the full legal name of the company?
    - What industry or niche do they primarily operate in?
    - Where is the company headquartered (city & country)?
    - What is the current estimated employee count?
    - What is the company website URL as mentioned in the LinkedIn Data?
    - Name of the individual
    - Their job title or designation
    - Are they likely a decision-maker (e.g., manager, VP, director, CXO)?
    - Are they hiring for roles that suggest growth, scaling, or specific operational challenges? If yes, mention the roles.
    - Based on the company's industry, employee size, and growth stage, does it align with the Ideal Customer Profile (ICP) outlined in the ICP definition? Provide a brief rationale for your assessment.
    - Are they likely to have the budget and maturity to engage with our service/product?
    - Have they posted or reshared any content that shows their pain points or areas of focus? Summarize relevant content if available.
    - Mention any known external tools or platforms the company uses (e.g., CRMs, marketing automation, cloud platforms, AI tools).
    - Can you derive a clear value proposition we might be able to offer, based on their context?

    Return your answer as a structured JSON list of question-answer objects.
    Only include questions for which you can find a clear answer in the provided data.
    """
    return system_prompt, user_prompt
