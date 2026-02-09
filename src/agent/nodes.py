from __future__ import annotations

import logging
import os
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
    run_command,
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

# TODO: Test the distribution thoroughly
def initialize_debugging_scores(state: DebuggingState) -> Dict[str, Any]:
    """
    Explicitly initializes the confidence_score for all nodes in the call graph.
    This runs once before the debugging cycle begins.
    Confidence scores are normalized so that their sum across all nodes is 1.0.
    """
    logger.info("initialize_debugging_scores: start")
    call_graph = state.get("call_graph")
    if not call_graph or "nodes" not in call_graph:
        return {"call_graph": call_graph}

    # 1. Collect raw scores with a small floor to ensure coverage
    nodes = call_graph["nodes"]
    raw_scores = []
    for node in nodes:
        # Use suspiciousness as the basis, with a small floor
        raw_score = max(0.01, node.get("suspiciousness", 0.0))
        raw_scores.append(raw_score)

    # 2. Normalize so that sum(scores) == 1.0
    total_score = sum(raw_scores)
    if total_score > 0:
        for node, score in zip(nodes, raw_scores):
            node["confidence_score"] = score / total_score
    else:
        # Fallback: equal distribution
        uniform_score = 1.0 / len(nodes)
        for node in nodes:
            node["confidence_score"] = uniform_score
    
    logger.info(f"Initialized and normalized confidence scores for {len(nodes)} nodes.")
    return {"call_graph": call_graph}


def select_target_node(state: DebuggingState) -> Dict[str, Any]:
    logger.info("select_target_node: start")
    call_graph = state.get("call_graph")
    if not call_graph or "nodes" not in call_graph:
        raise ValueError("Call graph is missing or invalid")

    # Find node with highest confidence_score 
    nodes = call_graph["nodes"]
    if not nodes:
        raise ValueError("No nodes in call graph")

    # Only consider nodes that have code (file, start_line)
    valid_nodes = [
        n
        for n in nodes
        if n.get("file") and n.get("start_line")
    ]
    
    if not valid_nodes:
        raise ValueError("No valid nodes for inspection found in call graph")

    # Highest confidence_score is the target
    target = max(valid_nodes, key=lambda x: x.get("confidence_score", 0.0))
    
    current_score = target.get("confidence_score", 0.0)
    logger.info(f"Selected target node: {target['fqn']} (confidence_score: {current_score:.4f})")

    history_entry = {
        "node": "select_target_node",
        "timestamp": time.time(),
        "data": {
            "selected_target": target["fqn"],
            "score": current_score,
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
    
    container_id = state.get("container_id")
    container_workspace = state.get("container_workspace")
    host_workspace = state.get("host_workspace")

    # 1. Determine which test to run
    test_files = find_test_files_for_node(call_graph, target_fqn)
    
    if test_files:
        # If running in Docker, we need to map host paths back to container paths
        if container_id and container_workspace and host_workspace:
            mapped_test_files = []
            for tf in test_files:
                if tf.startswith(host_workspace):
                    mapped_test_files.append(tf.replace(host_workspace, container_workspace))
                else:
                    mapped_test_files.append(tf)
            test_files = mapped_test_files

        # Use a robust pytest command (python -m pytest)
        test_command = f"python3 -m pytest {' '.join(test_files)}"
        
        # If we have a project-specific venv in the container, try to use its python
        if container_id and container_workspace:
            # Check if env/bin/python exists in the container 
            test_command = f"if [ -f env/bin/python ]; then ./env/bin/python -m pytest {' '.join(test_files)}; else python3 -m pytest {' '.join(test_files)}; fi"
            
        logger.info(f"Inferred test command: {test_command}")
    else:
        # Fallback to configured or default test command
        test_command = state.get("test_command", "python3 -m pytest")
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

    # 3. Run execution (local or docker)
    try:
        if container_id:
            result = run_command(test_command, container_id=container_id, workdir=container_workspace)
        else:
            result = run_command(test_command)
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
    """
    Updates the suspiciousness score using a formal Bayesian update based on execution artifacts.
    """
    logger.info("update_suspiciousness_and_reflect: start (Bayesian Mode)")
    target_fqn = state.get("target_node")
    call_graph = state.get("call_graph")
    execution_result = str(state.get("execution_result", ""))
    
    node = next((n for n in call_graph["nodes"] if n["fqn"] == target_fqn), None)
    if not node:
        raise ValueError(f"Node {target_fqn} not found")

    # Bayesian Prior: use existing confidence_score (guaranteed to exist now)
    prior = node.get("confidence_score", 0.5)
    
    # 1. Detect Formal Signals
    heartbeat_msg = f"--- INSPECTION_START: {target_fqn} ---"
    is_covered = heartbeat_msg in execution_result
    
    # Check if an AssertionError occurred in the target file
    target_file_short = os.path.basename(node.get("file", ""))
    has_target_assertion = (
        "AssertionError" in execution_result and target_file_short in execution_result
    )

    is_failure = "Exit Code: 0" not in execution_result

    # 2. Determine Likelihoods P(E|Buggy) and P(E|NotBuggy)
    if has_target_assertion:
        # Case A: Formal proof of failure in target
        p_e_given_buggy = 0.95
        p_e_given_not_buggy = 0.05
        outcome = "TARGET_ASSERTION_FAILED"
    elif is_covered and not is_failure:
        # Case B: Formal proof of success (covered and passed)
        p_e_given_buggy = 0.10
        p_e_given_not_buggy = 0.90
        outcome = "COVERED_AND_PASSED"
    elif is_failure and not has_target_assertion:
        # Case C: Collateral failure (test failed but not obviously here)
        p_e_given_buggy = 0.60
        p_e_given_not_buggy = 0.40
        outcome = "COLLATERAL_FAILURE"
    elif not is_covered:
        # Case D: Uninformative (not covered)
        p_e_given_buggy = 0.50
        p_e_given_not_buggy = 0.50
        outcome = "NO_COVERAGE"
    else:
        # Default/Fallback
        p_e_given_buggy = 0.50
        p_e_given_not_buggy = 0.50
        outcome = "INCONCLUSIVE"

    # 3. Apply Bayes Theorem
    # P(B|E) = (P(E|B) * P(B)) / (P(E|B) * P(B) + P(E|~B) * P(~B))
    denominator = (p_e_given_buggy * prior) + (p_e_given_not_buggy * (1 - prior))
    if denominator == 0:
        posterior = prior
    else:
        posterior = (p_e_given_buggy * prior) / denominator
    
    # Clamp and update confidence_score (do not touch suspiciousness)
    node["confidence_score"] = max(0.01, min(0.99, posterior))
    
    # 4. LLM Reflection (Now secondary, for reasoning only)
    source_code = get_function_source(node)
    patch = state.get("inspection_patch")
    prompt = build_debugging_reflection_prompt(
        target_fqn, source_code, patch, execution_result
    )
    llm = get_default_llm_connector()
    reflection = llm.generate(prompt)

    logger.info(
        f"Bayesian Update for {target_fqn}: outcome={outcome}, "
        f"prior={prior:.3f} -> posterior={node['confidence_score']:.3f}"
    )

    history_entry = {
        "node": "update_suspiciousness_and_reflect",
        "timestamp": time.time(),
        "data": {
            "target_node": target_fqn,
            "outcome": outcome,
            "is_covered": is_covered,
            "has_target_assertion": has_target_assertion,
            "old_score": prior,
            "new_score": node["confidence_score"],
            "reflection": reflection
        }
    }
    
    current_history = state.get("history", []) + [history_entry]
    save_history(current_history)

    return {
        "call_graph": call_graph,
        "reflection": reflection,
        "history": [history_entry],
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
    }


if __name__ == "__main__":
    import json
    
    # Path to a demo call graph
    demo_path = "artifacts/demo_call_graph_dynamic_with_suspiciousness.json"
    
    try:
        with open(demo_path, "r") as f:
            cg_data = json.load(f)
        
        # Wrap in mock state
        mock_state: DebuggingState = {"call_graph": cg_data}
        
        # Run initialization
        result = initialize_debugging_scores(mock_state)
        updated_cg = result["call_graph"]
        
        print("\n" + "="*80)
        print(f"{'Node FQN':<50} | {'Suspiciousness':<12} | {'Confidence'}")
        print("-" * 80)
        
        total_conf = 0
        for node in updated_cg["nodes"]:
            susp = node.get("suspiciousness", 0.0)
            conf = node.get("confidence_score", 0.0)
            total_conf += conf
            print(f"{node['fqn'][:50]:<50} | {susp:<12.4f} | {conf:.4f}")
            
        print("-" * 80)
        print(f"{'TOTAL':<50} | {'':<12} | {total_conf:.4f}")
        print("="*80)


        # Print top 5 nodes by confidence score
        print("\nTop 5 nodes by confidence score (sorted by confidence score in descending order):")
        for node in sorted(updated_cg["nodes"], key=lambda x: x.get("confidence_score", 0.0), reverse=True)[:5]:
            conf = node.get("confidence_score", 0.0)
            print(f"{node['fqn'][:50]:<50} | {conf:.4f}")
        print("="*80)
        
    except FileNotFoundError:
        print(f"Error: Demo file not found at {demo_path}")
    except Exception as e:
        print(f"An error occurred: {e}")