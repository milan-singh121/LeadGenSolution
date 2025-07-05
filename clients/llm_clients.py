"""
This script contains the client for interacting with LLMs,
specifically Anthropic's Claude model via AWS Bedrock.
"""

import os
import sys
import logging
from anthropic import AnthropicBedrock

app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir))  # already project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import AWS credentials from the central config file
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    A client to interact with the Claude AI model via Anthropic's AWS Bedrock service.
    """

    def __init__(self):
        """
        Initializes the Anthropic Bedrock client.
        """
        if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION]):
            raise ValueError("AWS credentials and region must be set in the config.")

        try:
            self.client = AnthropicBedrock(
                aws_access_key=AWS_ACCESS_KEY,
                aws_secret_key=AWS_SECRET_KEY,
                aws_region=AWS_REGION,
            )
            logger.info("Anthropic Bedrock client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic Bedrock client: {e}")
            raise

    def get_structured_response(
        self, system_prompt: str, user_prompt: str
    ) -> (str, float):
        """
        Sends a prompt to the Claude 3.5 Sonnet model and returns the response and cost.

        Args:
            system_prompt (str): The system prompt to guide the AI's behavior.
            user_prompt (str): The user's prompt or question.

        Returns:
            A tuple containing the text content of the AI's response and the calculated cost.
        """
        try:
            response = self.client.messages.create(
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0,
                max_tokens=4096,
                timeout=180,
            )

            # Log usage and cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            input_cost = (input_tokens / 1_000_000) * 3
            output_cost = (output_tokens / 1_000_000) * 15
            total_cost = input_cost + output_cost

            logger.info(
                f"Claude 3.5 Sonnet call stats: "
                f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, "
                f"Total Cost: ${total_cost:.5f}"
            )

            return response.content[0].text, total_cost

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            raise
