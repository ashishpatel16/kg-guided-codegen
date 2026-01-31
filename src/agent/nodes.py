from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.state import OneShotCodeGenState, DebuggingState
from src.agent.tools import (
    build_hypothesis_prompt,
    build_one_shot_prompt,
    extract_code,
    get_default_llm_connector,
    build_evidence_prompt,
    build_evidence_evaluation_prompt,
    build_reflection_prompt,
    get_function_source,
    apply_function_source,
    run_local_command,
    build_inspection_patch_prompt,
    build_debugging_reflection_prompt,
    find_test_files_for_node,
    save_history,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s][%(levelname)s] %(message)s'
)


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

    history_entry = {
        "node": "generate_code",
        "timestamp": time.time(),
        "data": {
            "problem": problem,
            "code_chars": len(code or ""),
            "duration_ms": duration_ms
        }
    }

    return {
        "generated_code": code,
        "messages": new_messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "history": [history_entry]
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

    history_entry = {
        "node": "generate_hypothesis",
        "timestamp": time.time(),
        "data": {
            "problem": problem,
            "hypothesis": hypothesis,
            "duration_ms": duration_ms
        }
    }

    return {
        "hypothesis": hypothesis,
        "messages": new_messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "history": [history_entry]
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

    history_entry = {
        "node": "generate_evidence",
        "timestamp": time.time(),
        "data": {
            "hypothesis": hypothesis,
            "evidence": evidence,
            "duration_ms": duration_ms
        }
    }

    return {
        "evidence": evidence,
        "messages": messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "history": [history_entry]
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

    history_entry = {
        "node": "evaluate_evidence",
        "timestamp": time.time(),
        "data": {
            "evidence": evidence,
            "evaluation": evidence_evaluation,
            "duration_ms": duration_ms
        }
    }

    return {
        "evidence_evaluation": evidence_evaluation,
        "messages": messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "history": [history_entry]
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

    history_entry = {
        "node": "generate_reflection",
        "timestamp": time.time(),
        "data": {
            "hypothesis": hypothesis,
            "reflection": reflection,
            "duration_ms": duration_ms
        }
    }

    # Save history for one-shot too
    current_history = state.get("history", []) + [history_entry]
    save_history(current_history, file_path="artifacts/one_shot_history.json")

    return {
        "reflection": reflection,
        "messages": messages,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "history": [history_entry]
    }


def select_target_node(state: DebuggingState) -> Dict[str, Any]:
    logger.info("select_target_node: start")
    call_graph = state.get("call_graph")
    if not call_graph or "nodes" not in call_graph:
        raise ValueError("Call graph is missing or invalid")

    # Find node with highest suspiciousness
    nodes = call_graph["nodes"]
    if not nodes:
        raise ValueError("No nodes in call graph")

    # Only consider nodes that have code (file, start_line)
    valid_nodes = [
        n
        for n in nodes
        if n.get("file") and n.get("start_line") and n.get("suspiciousness") is not None
    ]
    if not valid_nodes:
        raise ValueError("No valid nodes for inspection found in call graph")

    target = max(valid_nodes, key=lambda x: x["suspiciousness"])
    logger.info(f"Selected target node: {target['fqn']} (suspiciousness: {target['suspiciousness']})")

    history_entry = {
        "node": "select_target_node",
        "timestamp": time.time(),
        "data": {
            "selected_target": target["fqn"],
            "suspiciousness": target["suspiciousness"],
            "file": target["file"]
        }
    }

    return {
        "target_node": target["fqn"],
        "history": [history_entry]
    }


def generate_inspection_patch(state: DebuggingState) -> Dict[str, Any]:
    target_fqn = state.get("target_node")
    call_graph = state.get("call_graph")
    node = next((n for n in call_graph["nodes"] if n["fqn"] == target_fqn), None)

    print(f"DEBUG: generate_inspection_patch for {target_fqn}")
    if node:
        print(f"DEBUG: Node found in call graph. File: {node.get('file')}")
    else:
        print(f"DEBUG: Node NOT FOUND in call graph!")

    if not node:
        raise ValueError(f"Target node {target_fqn} not found in call graph")

    source_code = get_function_source(node)
    if not source_code:
        raise ValueError(f"Could not fetch source code for {target_fqn}")

    logger.info(f"generate_inspection_patch: generating for {target_fqn}")
    prompt = build_inspection_patch_prompt(target_fqn, source_code)
    llm = get_default_llm_connector()
    raw_patch = llm.generate(prompt)
    patch = extract_code(raw_patch)

    logger.info(f"generate_inspection_patch: done (patch=\n{patch})")

    history_entry = {
        "node": "generate_inspection_patch",
        "timestamp": time.time(),
        "data": {
            "target_node": target_fqn,
            "patch": patch
        }
    }

    return {
        "inspection_patch": patch,
        "original_source": source_code,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "history": [history_entry]
    }


def execute_inspection(state: DebuggingState) -> Dict[str, Any]:
    patch = state.get("inspection_patch")
    original_source = state.get("original_source")
    target_fqn = state.get("target_node")
    call_graph = state.get("call_graph")
    
    # 1. Determine which test to run
    test_files = find_test_files_for_node(call_graph, target_fqn)
    if test_files:
        # Run only the relevant test files
        test_command = f"pytest {' '.join(test_files)}"
        logger.info(f"Inferred test command: {test_command}")
    else:
        # Fallback to configured or default test command
        test_command = state.get("test_command", "pytest")
        logger.warning(f"No specific test files found for {target_fqn}, using fallback: {test_command}")

    node = next((n for n in call_graph["nodes"] if n["fqn"] == target_fqn), None)
    if not node:
        raise ValueError(f"Target node {target_fqn} not found in call graph")

    if not patch:
        raise ValueError("No inspection patch to execute")

    logger.info(f"execute_inspection: applying patch to {target_fqn} and running tests")

    # 2. Apply patch
    if not apply_function_source(node, patch):
        logger.error(f"execute_inspection: failed to apply patch to {target_fqn}")
        return {"execution_result": "Error: Failed to apply patch to file."}

    # 3. Run local execution
    try:
        result = run_local_command(test_command)
    finally:
        # 4. Restore original source
        logger.info(f"execute_inspection: restoring original source for {target_fqn}")
        apply_function_source(node, original_source)

    history_entry = {
        "node": "execute_inspection",
        "timestamp": time.time(),
        "data": {
            "target_node": target_fqn,
            "test_command": test_command,
            "result": result
        }
    }

    return {
        "execution_result": result,
        "history": [history_entry]
    }


def update_suspiciousness_and_reflect(state: DebuggingState) -> Dict[str, Any]:
    logger.info("update_suspiciousness_and_reflect: start")
    target_fqn = state.get("target_node")
    call_graph = state.get("call_graph")
    execution_result = state.get("execution_result")
    score_delta = state.get("score_delta", 0.3)

    node = next((n for n in call_graph["nodes"] if n["fqn"] == target_fqn), None)
    source_code = get_function_source(node)
    patch = state.get("inspection_patch")

    prompt = build_debugging_reflection_prompt(
        target_fqn, source_code, patch, execution_result
    )
    llm = get_default_llm_connector()
    reflection = llm.generate(prompt)

    # Parse the decision from reflection
    is_buggy = "DECISION: BUGGY" in reflection or "RESULT: BUGGY" in reflection
    is_not_buggy = "RESULT: NOT_BUGGY" in reflection or "DECISION: NOT_BUGGY" in reflection

    # Strong heuristic check: if the LLM blames another function, it's not buggy
    blames_dependency = (
        "root cause is in" in reflection.lower() 
        or "bug is in" in reflection.lower()
        or "dependency is buggy" in reflection.lower()
    ) and target_fqn.lower() not in reflection.lower().split("bug is in")[-1]

    old_score = node["suspiciousness"]
    
    if is_buggy and not is_not_buggy and not blames_dependency:
        node["suspiciousness"] += score_delta
        logger.info(f"Node {target_fqn} suspiciousness increased from {old_score} to {node['suspiciousness']}")
    elif is_not_buggy or blames_dependency:
        node["suspiciousness"] -= score_delta
        logger.info(f"Node {target_fqn} suspiciousness decreased from {old_score} to {node['suspiciousness']}")
    else:
        # Fallback heuristic
        is_buggy_heuristic = (
            "likely buggy" in reflection.lower()
            or "supports the hypothesis" in reflection.lower()
        )
        if is_buggy_heuristic and not blames_dependency:
            node["suspiciousness"] += score_delta
        else:
            node["suspiciousness"] -= score_delta
        logger.info(f"Node {target_fqn} suspiciousness updated via heuristic from {old_score} to {node['suspiciousness']}")

    history_entry = {
        "node": "update_suspiciousness_and_reflect",
        "timestamp": time.time(),
        "data": {
            "target_node": target_fqn,
            "reflection": reflection,
            "old_score": old_score,
            "new_score": node["suspiciousness"],
            "is_buggy_detected": is_buggy
        }
    }

    # Update history in state
    current_history = state.get("history", []) + [history_entry]
    save_history(current_history)

    return {
        "call_graph": call_graph,
        "reflection": reflection,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "history": [history_entry]
    }
