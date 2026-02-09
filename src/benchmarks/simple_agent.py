import json
import logging
import os
import argparse
from typing import List, Dict, Any
import re

from src.agent.tools import get_function_source, get_default_llm_connector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s][%(levelname)s] %(message)s'
)

INFER_PROBLEM_PROMPT_TEMPLATE = """
You are an expert debugger. You are given a list of suspicious code snippets and the tests that exercise them.
Based on these snippets and tests, infer what the bug / problem might be.

Suspicious Candidates:
{candidates_formatted}

Failing Tests (if any):
{tests_formatted}

Provide a concise summary of the suspected problem.
Output ONLY the summary.
"""

RANK_BUG_CANDIDATES_PROMPT_TEMPLATE = """
You are an expert debugger. You are given a failing test case / problem description and a list of code snippets (functions/methods) that might contain the bug.
Your task is to rank these code snippets in descending order of how likely each one is to contain the root cause of the bug.

Problem Description / Failing Test Case:
{problem_description}

Candidates:
{candidates_formatted}

Rules:
1. Analyze the problem description and the provided code carefully.
2. For each candidate, consider if its logic could be responsible for the reported failure.
3. Provide a brief reasoning for each candidate.
4. Output the final ranking as a JSON list of FQNs in descending order of bug likelihood.

Format your response as follows:
Reasoning:
<your reasoning for each candidate>

Ranking:
```json
[
  "fqn.of.most.likely.node",
  "fqn.of.second.most.likely.node",
  ...
]
```
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
def visualize_rankings(rankings: List[Dict], output_path: str = f"artifacts/ranking_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"):
    """Visualizes the stability of rankings across multiple attempts."""
    if not rankings:
        return

    # 1. Identify all candidates that were supposed to be ranked, candidates from the first attempt are used as the baseline
    all_candidates = [c["fqn"] for c in rankings[0]["top_k_candidates"]]
    num_attempts = len(rankings)
    num_candidates = len(all_candidates)

    # 2. Prepare a matrix for heatmap: (candidate, attempt) -> rank
    # Use 0 for "not ranked" or a large number. Let's use 1-indexed ranks.
    rank_matrix = np.zeros((num_candidates, num_attempts))

    for attempt_idx, result in enumerate(rankings):
        ranking = result.get("extracted_ranking", [])
        for cand_idx, cand_fqn in enumerate(all_candidates):
            try:
                # Rank is 1-indexed. If not found, use a "bad" rank (num_candidates + 1)
                rank = ranking.index(cand_fqn) + 1
            except ValueError:
                rank = num_candidates + 1
            rank_matrix[cand_idx, attempt_idx] = rank

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rank_matrix, cmap="YlOrRd_r")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(num_attempts))
    ax.set_yticks(np.arange(num_candidates))
    ax.set_xticklabels([f"Attempt {i+1}" for i in range(num_attempts)])
    ax.set_yticklabels([fqn.split(".")[-1] for fqn in all_candidates]) # Shorten FQNs

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(num_candidates):
        for j in range(num_attempts):
            val = rank_matrix[i, j]
            label = int(val) if val <= num_candidates else "N/A"
            ax.text(j, i, label, ha="center", va="center", color="black")

    ax.set_title("Bug Ranking Stability Across LLM Attempts\n(Lower number = More likely to be the bug)")
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Ranking stability visualization saved to {output_path}")

def extract_json_ranking(text: str) -> List[str]:
    """Extracts the JSON ranking from the LLM response."""
    # Look for a JSON list in the text
    pattern = r"```json\s*(\[[\s\S]*?\])\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Fallback: look for any JSON-like list
    pattern_fallback = r"(\[[\s\S]*?\])"
    matches = re.finditer(pattern_fallback, text)
    for m in reversed(list(matches)):
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return data
        except json.JSONDecodeError:
            continue
            
    return []

def infer_problem(llm, candidates: List[Dict], call_graph: Dict, repo_root: str) -> str:
    """Infers the problem description from suspicious candidates and related tests."""
    logger.info("Inferring problem description from candidates and tests...")
    
    # 1. Gather tests that call these candidates
    related_tests = []
    test_fqns = set()
    
    for cand in candidates:
        # Find callers of this candidate
        callers = [
            edge["source"] 
            for edge in call_graph.get("edges", []) 
            if edge["target"] == cand["fqn"]
        ]
        
        for caller_fqn in callers:
            if caller_fqn in test_fqns:
                continue
            
            caller_node = next(
                (n for n in call_graph.get("nodes", []) if n["fqn"] == caller_fqn), 
                None
            )
            
            if caller_node:
                file_path = caller_node.get("file", "")
                if "test" in file_path.lower() or "test" in caller_fqn.lower():
                    # This looks like a test
                    source = get_function_source(caller_node)
                    if source:
                        related_tests.append({
                            "fqn": caller_fqn,
                            "source": source
                        })
                        test_fqns.add(caller_fqn)
                        if len(related_tests) >= 3: # Limit to 3 tests for brevity
                            break
        if len(related_tests) >= 3:
            break

    # 2. Format for prompt
    candidates_formatted = ""
    for idx, cand in enumerate(candidates[:k]): # Limit to top 3 for inference
        candidates_formatted += f"--- Candidate {idx+1}: {cand['fqn']} ---\n"
        candidates_formatted += f"Code:\n```python\n{cand['source']}\n```\n\n"
        
    tests_formatted = ""
    for idx, test in enumerate(related_tests):
        tests_formatted += f"--- Test {idx+1}: {test['fqn']} ---\n"
        tests_formatted += f"Code:\n```python\n{test['source']}\n```\n\n"

    if not tests_formatted:
        tests_formatted = "No specific failing tests identified in the call graph."

    prompt = INFER_PROBLEM_PROMPT_TEMPLATE.format(
        candidates_formatted=candidates_formatted,
        tests_formatted=tests_formatted
    )

    print(f"Prompt: {prompt}")
    
    inferred_problem = llm.generate(prompt)
    logger.info(f"Inferred Problem: {inferred_problem}")
    return inferred_problem

def fix_call_graph_paths(call_graph: Dict, repo_root: str):
    """Adjusts file paths in the call graph nodes to be absolute and existing on the local machine."""
    nodes = call_graph.get("nodes", [])
    for node in nodes:
        original_file = node.get("file", "")
        if not original_file:
            continue
        
        # If the path already exists, just make it absolute if it's not
        if os.path.exists(original_file):
            if not os.path.isabs(original_file):
                node["file"] = os.path.abspath(original_file)
            continue
            
        # If it doesn't exist, try making it relative to repo_root by stripping leading path components
        parts = original_file.split("/")
        found = False
        for i in range(len(parts)):
            candidate_path = os.path.join(repo_root, *parts[i:])
            if os.path.exists(candidate_path):
                node["file"] = os.path.abspath(candidate_path)
                found = True
                break
        
        if not found:
            logger.debug(f"Could not resolve path for {node.get('fqn')}: {original_file}")

def run_benchmark(
    call_graph_path: str,
    problem_description: str = None,
    k: int = 7,
    repo_root: str = ".",
    output_path: str = "benchmark_results.json"
):
    logger.info(f"Loading call graph from {call_graph_path} with k={k}")
    with open(call_graph_path, "r") as f:
        call_graph = json.load(f)

    # Fix paths globally in the call graph
    fix_call_graph_paths(call_graph, repo_root)

    nodes = call_graph.get("nodes", [])
    if not nodes:
        logger.error("No nodes found in call graph")
        return

    # Sort nodes by suspiciousness in descending order
    # Handle cases where suspiciousness might be missing
    sorted_nodes = sorted(
        nodes, 
        key=lambda x: x.get("suspiciousness", 0.0), 
        reverse=True
    )

    # Pick top k nodes that have file and start_line
    top_k_candidates = []
    for node in sorted_nodes:
        if len(top_k_candidates) >= k:
            break
        
        source = get_function_source(node)
        if source:
            top_k_candidates.append({
                "fqn": node["fqn"],
                "source": source,
                "suspiciousness": node.get("suspiciousness", 0.0)
            })
        else:
            logger.warning(f"Could not fetch source for {node['fqn']} at {node.get('file')}")

    if not top_k_candidates:
        logger.error("No valid candidates found with source code")
        return

    logger.info(f"Found {len(top_k_candidates)} candidates for ranking")

    llm = get_default_llm_connector()

    # Infer problem if not provided
    if not problem_description or problem_description.strip() == "":
        problem_description = infer_problem(llm, top_k_candidates, call_graph, repo_root)

    # Format candidates for the prompt
    candidates_formatted = ""
    for idx, cand in enumerate(top_k_candidates):
        candidates_formatted += f"--- Candidate {idx+1}: {cand['fqn']} ---\n"
        candidates_formatted += f"Suspiciousness Score: {cand['suspiciousness']}\n"
        candidates_formatted += f"Code:\n```python\n{cand['source']}\n```\n\n"

    prompt = RANK_BUG_CANDIDATES_PROMPT_TEMPLATE.format(
        problem_description=problem_description,
        candidates_formatted=candidates_formatted
    )

    logger.info("Calling LLM for ranking...")
    llm = get_default_llm_connector()
    response = llm.generate(prompt)

    ranking = extract_json_ranking(response)
    
    result = {
        "call_graph_path": call_graph_path,
        "problem_description": problem_description,
        "k": k,
        "top_k_candidates": [
            {"fqn": c["fqn"], "suspiciousness": c["suspiciousness"]} 
            for c in top_k_candidates
        ],
        "llm_response": response,
        "extracted_ranking": ranking
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Benchmark result saved to {output_path}")
    print(f"\nFinal Ranking: {ranking}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple bug ranking benchmark.")
    parser.add_argument("call_graph", help="Path to the dynamic call graph JSON file")
    parser.add_argument("--problem", help="Problem description or path to a file containing it. If not provided, it will be inferred.")
    parser.add_argument("--k", type=int, default=7, help="Number of top suspicious nodes to rank")
    parser.add_argument("--repo_root", default=".", help="Root directory of the repository")
    parser.add_argument("--output", default="artifacts/benchmark_simple_agent.json", help="Output JSON path")

    args = parser.parse_args()

    # If problem is a path to a file, read it
    problem_text = args.problem
    if problem_text and os.path.exists(problem_text):
        with open(problem_text, "r") as f:
            problem_text = f.read()

    rankings = []
    k = 5
    for i in range(k):
        result =run_benchmark(
            call_graph_path=args.call_graph,
            problem_description=problem_text,
            k=args.k,
            repo_root=args.repo_root,
            output_path=args.output
        )
        rankings.append(result)

    

    print(f"Comparison of all attempts")
    for i, ranking in enumerate(rankings):
        print(f"Attempt {i+1}: {ranking['extracted_ranking']}")
    
    visualize_rankings(rankings)
