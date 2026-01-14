from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from agent.state import OneShotCodeGenState
from agent.tools import (
    build_hypothesis_prompt,
    build_one_shot_prompt,
    extract_code,
    get_default_llm_connector,
    build_evidence_prompt,
    build_evidence_evaluation_prompt,
    build_reflection_prompt,
)

logger = logging.getLogger(__name__)


def generate_code(state: OneShotCodeGenState) -> Dict[str, Any]:
    start_t = time.time()

    problem = (state.get("problem") or "").strip()
    messages = state.get("messages") or []

    if not problem:
        raise ValueError("Problem is required")

    if not messages:
        raise ValueError("Messages are required")

    logger.info(
        "generate_code: start (problem_chars=%s, has_messages=%s, prior_llm_calls=%s)",
        len(problem),
        len(messages),
        state.get("llm_calls"),
    )

    prompt = build_one_shot_prompt(problem)
    llm = get_default_llm_connector()
    raw = llm.generate(prompt)

    code = extract_code(raw)

    new_messages: List[Any] = []
    if not messages and problem:
        new_messages.append(HumanMessage(content=problem))
    new_messages.append(AIMessage(content=raw))

    duration_ms = int((time.time() - start_t) * 1000)
    logger.info(
        "generate_code: done (duration_ms=%s, raw_chars=%s, code_chars=%s)",
        duration_ms,
        len(raw or ""),
        len(code or ""),
    )

    return {
        "generated_code": code,
        "messages": new_messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
    }


def generate_hypothesis(state: OneShotCodeGenState) -> Dict[str, Any]:
    start_t = time.time()

    problem = (state.get("problem") or "").strip()
    messages = state.get("messages") or []
    reflection = (state.get("reflection") or "").strip()

    if not problem:
        raise ValueError("Problem is required")

    logger.info(
        "generate_hypothesis: start (problem_chars=%s, has_messages=%s, prior_llm_calls=%s, reflection=%s)",
        len(problem),
        len(messages),
        state.get("llm_calls"),
    )

    prompt = build_hypothesis_prompt(problem)
    llm = get_default_llm_connector()
    hypothesis = llm.generate(prompt)

    new_messages: List[Any] = []
    if not messages and problem:
        new_messages.append(HumanMessage(content=problem))
    new_messages.append(AIMessage(content=hypothesis))

    duration_ms = int((time.time() - start_t) * 1000)
    logger.info(
        f"generate_hypothesis: done (hypothesis={hypothesis}, duration_ms={duration_ms})",
    )

    return {
        "hypothesis": hypothesis,
        "messages": new_messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
    }


def generate_evidence(state: OneShotCodeGenState) -> Dict[str, Any]:
    logger.info(
        f"generate_evidence: start (problem_chars={len(state.get('problem') or '')}, hypothesis={state.get('hypothesis')})"
    )
    start_t = time.time()

    problem = (state.get("problem") or "").strip()
    hypothesis = (state.get("hypothesis") or "").strip()
    code = (state.get("generated_code") or "").strip()
    messages = state.get("messages") or []

    if not problem or not hypothesis or not code:
        raise ValueError("Problem, hypothesis, and code are required")

    prompt = build_evidence_prompt(problem, hypothesis, code)
    llm = get_default_llm_connector()
    evidence = llm.generate(prompt)

    duration_ms = int((time.time() - start_t) * 1000)
    logger.info(
        f"generate_evidence: done (evidence={evidence}, duration_ms={duration_ms})"
    )
    return {
        "evidence": evidence,
        "messages": messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
    }


def evaluate_evidence(state: OneShotCodeGenState) -> Dict[str, Any]:
    logger.info(
        f"evaluate_evidence: start (problem_chars={len(state.get('problem') or '')}, hypothesis={state.get('hypothesis')})"
    )
    start_t = time.time()

    problem = (state.get("problem") or "").strip()
    hypothesis = (state.get("hypothesis") or "").strip()
    evidence = (state.get("evidence") or "").strip()
    code = (state.get("generated_code") or "").strip()
    messages = state.get("messages") or []

    if not problem or not hypothesis or not evidence or not code:
        raise ValueError("Problem, hypothesis, evidence, and code are required")

    prompt = build_evidence_evaluation_prompt(problem, hypothesis, evidence, code)
    llm = get_default_llm_connector()
    evidence_evaluation = llm.generate(prompt)

    duration_ms = int((time.time() - start_t) * 1000)
    logger.info(
        f"evaluate_evidence: done (evidence_evaluation={evidence_evaluation}, duration_ms={duration_ms})"
    )
    return {
        "evidence_evaluation": evidence_evaluation,
        "messages": messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
    }


def generate_reflection(state: OneShotCodeGenState) -> Dict[str, Any]:
    logger.info(
        f"generate_reflection: start (problem_chars={len(state.get('problem') or '')}, hypothesis={state.get('hypothesis')})"
    )
    start_t = time.time()

    problem = (state.get("problem") or "").strip()
    hypothesis = (state.get("hypothesis") or "").strip()
    evidence = (state.get("evidence") or "").strip()
    evidence_evaluation = float(state.get("evidence_evaluation") or 0)
    code = (state.get("generated_code") or "").strip()
    messages = state.get("messages") or []

    if not problem or not hypothesis or not code:
        raise ValueError("Problem, hypothesis, and code are required")

    prompt = build_reflection_prompt(problem, hypothesis, code)
    llm = get_default_llm_connector()
    reflection = llm.generate(prompt)

    duration_ms = int((time.time() - start_t) * 1000)
    logger.info(
        f"generate_reflection: done (reflection={reflection}, duration_ms={duration_ms})"
    )
    return {
        "reflection": reflection,
        "messages": messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
    }
