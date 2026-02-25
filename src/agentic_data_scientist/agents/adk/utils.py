"""
Utility functions and configurations for ADK agents.

This module provides model configuration, helper functions, and shared settings
for the ADK agent system.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types


load_dotenv()

logger = logging.getLogger(__name__)


# Model configuration
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-pro")
REVIEW_MODEL_NAME = os.getenv("REVIEW_MODEL", "gemini/gemini-2.5-pro")
CODING_MODEL_NAME = os.getenv("CODING_MODEL", "claude-sonnet-4-5-20250929")

logger.info(f"[AgenticDS] DEFAULT_MODEL={DEFAULT_MODEL_NAME}")
logger.info(f"[AgenticDS] REVIEW_MODEL={REVIEW_MODEL_NAME}")
logger.info(f"[AgenticDS] CODING_MODEL={CODING_MODEL_NAME}")

# Configure LiteLLM for Nexus (LiteLLM Proxy)
NEXUS_URL = os.getenv("NEXUS_URL", "https://nexus-master.lmndstaging.com")
NEXUS_API_KEY = os.getenv("LITELLM_PROXY_API_KEY", "sk-12345")
NEXUS_API_BASE = f"{NEXUS_URL}/v1"

# Export for use in event compression
__all__ = [
    'DEFAULT_MODEL',
    'REVIEW_MODEL',
    'DEFAULT_MODEL_NAME',
    'REVIEW_MODEL_NAME',  # Export model name strings
    'NEXUS_API_KEY',
    'NEXUS_API_BASE',
    'get_generate_content_config',
    'exit_loop_simple',
    'is_network_disabled',
]

# Log Nexus configuration
logger.info(f"[AgenticDS] Nexus API base: {NEXUS_API_BASE}")

# Create LiteLLM model instances
# LiteLLM requires litellm_proxy/ prefix to route through Nexus proxy
DEFAULT_MODEL = LiteLlm(
    model=f"litellm_proxy/{DEFAULT_MODEL_NAME}",
    num_retries=10,
    timeout=60,
    api_base=NEXUS_API_BASE,
    api_key=NEXUS_API_KEY,
)

REVIEW_MODEL = LiteLlm(
    model=f"litellm_proxy/{REVIEW_MODEL_NAME}",
    num_retries=10,
    timeout=60,
    api_base=NEXUS_API_BASE,
    api_key=NEXUS_API_KEY,
)

# Language requirement (empty for English-only models)
LANGUAGE_REQUIREMENT = ""


def is_network_disabled() -> bool:
    """
    Check if network access is disabled via environment variable.

    Network access is enabled by default. Set DISABLE_NETWORK_ACCESS
    to "true" or "1" to disable network tools.

    Returns
    -------
    bool
        True if network access should be disabled, False otherwise
    """
    disable_network = os.getenv("DISABLE_NETWORK_ACCESS", "").lower()
    return disable_network in ("true", "1")


# DEPRECATED: Use review_confirmation agents instead
# This function is kept for backward compatibility but should not be used in new code.
# Loop exit decisions should be made by dedicated review_confirmation agents with
# structured output and callbacks, not by direct tool calls from review agents.
def exit_loop_simple(tool_context: ToolContext):
    """
    Exit the iterative loop when no further changes are needed.

    DEPRECATED: Use review_confirmation agents instead.

    This function is called by review agents to signal that the iterative
    process should end.

    Parameters
    ----------
    tool_context : ToolContext
        The tool execution context

    Returns
    -------
    dict
        Empty dictionary (tools should return JSON-serializable output)
    """
    tool_context.actions.escalate = True
    return {}


def get_generate_content_config(temperature: float = 0.0, output_tokens: Optional[int] = None):
    """
    Create a GenerateContentConfig with retry settings.

    Parameters
    ----------
    temperature : float, optional
        Sampling temperature (default: 0.0)
    output_tokens : int, optional
        Maximum output tokens

    Returns
    -------
    types.GenerateContentConfig
        Configuration for content generation
    """
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
        seed=42,
        max_output_tokens=output_tokens,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=50,
                initial_delay=1.0,
                max_delay=30,
                exp_base=1.5,
                jitter=0.5,
                http_status_codes=[429, 500, 502, 503, 504],
            )
        ),
    )
