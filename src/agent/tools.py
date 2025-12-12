from __future__ import annotations

import logging
import os
import re

from llm.connector import OllamaLLMConnector, RawOllamaConnector

logger = logging.getLogger(__name__)

from agent.prompts import DEFAULT_SYSTEM_INSTRUCTIONS, GENERATE_HYPOTHESIS_SYSTEM_INSTRUCTIONS, GENERATE_EVIDENCE_SYSTEM_INSTRUCTIONS, EVALUATE_EVIDENCE_SYSTEM_INSTRUCTIONS, GENERATE_REFLECTION_SYSTEM_INSTRUCTIONS


def configure_logging(level: str | None = None) -> None:
    """
    Basic logging setup for running locally.
    Configure via LOG_LEVEL env var (default: INFO).
    """
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, resolved, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_one_shot_prompt(problem: str) -> str:
    problem = (problem or "").strip()
    prompt = f"""{DEFAULT_SYSTEM_INSTRUCTIONS}

    Problem:
    {problem}

    """
    logger.debug("Built prompt (problem_chars=%s, prompt_chars=%s)", len(problem), len(prompt))
    return prompt


def build_hypothesis_prompt(problem: str) -> str:
    problem = (problem or "").strip()
    prompt = f"""{GENERATE_HYPOTHESIS_SYSTEM_INSTRUCTIONS}

    Problem:
    {problem}

    """
    return prompt

_FENCED_CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)\s*```", re.MULTILINE)


def extract_code(text: str) -> str:
    """
    Best-effort extraction of code from an LLM response.
    - If the model returns a fenced code block, return the first block's content.
    - Otherwise return the raw text trimmed.
    """
    text = (text or "").strip()
    if not text:
        return ""

    m = _FENCED_CODE_RE.search(text)
    if m:
        extracted = (m.group(1) or "").strip()
        logger.debug("Extracted fenced code (chars=%s)", len(extracted))
        return extracted

    logger.debug("No fenced block found; returning raw text (chars=%s)", len(text))
    return text


def get_default_llm_connector(model_name: str = "gemma3:12b", temperature: float = 0.5):
    llm = OllamaLLMConnector(model_name=model_name, temperature=temperature)
    logger.info("Created LLM connector (model=%s, temperature=%s)", model_name, temperature)
    return llm

def build_evidence_prompt(problem: str, hypothesis: str, code: str) -> str:
    problem = (problem or "").strip()
    hypothesis = (hypothesis or "").strip()
    code = (code or "").strip()
    prompt = f"""{GENERATE_EVIDENCE_SYSTEM_INSTRUCTIONS}

    Problem:
    {problem}

    Hypothesis:
    {hypothesis}

    Code:
    {code}

    """
    return prompt

def build_evidence_evaluation_prompt(problem: str, hypothesis: str, evidence: str, code: str) -> str:
    problem = (problem or "").strip()
    hypothesis = (hypothesis or "").strip()
    evidence = (evidence or "").strip()
    code = (code or "").strip()
    prompt = f"""{EVALUATE_EVIDENCE_SYSTEM_INSTRUCTIONS}

    Problem:
    {problem}

    Hypothesis:
    {hypothesis}

    Code:
    {code}

    Evidence:
    {evidence}

    """
    return prompt

def build_reflection_prompt(problem: str, hypothesis: str, code: str) -> str:
    problem = (problem or "").strip()
    hypothesis = (hypothesis or "").strip()
    code = (code or "").strip()

    prompt = f"""{GENERATE_REFLECTION_SYSTEM_INSTRUCTIONS}

    Problem:
    {problem}

    Hypothesis:
    {hypothesis}

    Code:
    {code}

    """
    return prompt