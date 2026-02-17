from __future__ import annotations

import ast
import difflib
import json
import logging
import os
import re
import subprocess
import docker

from src.llm.connector import (
    OllamaLLMConnector,
    RawOllamaConnector,
    GeminiLLMConnector,
)

from src.agent.prompts import (
    DEFAULT_SYSTEM_INSTRUCTIONS,
    GENERATE_HYPOTHESIS_SYSTEM_INSTRUCTIONS,
    GENERATE_EVIDENCE_SYSTEM_INSTRUCTIONS,
    EVALUATE_EVIDENCE_SYSTEM_INSTRUCTIONS,
    GENERATE_REFLECTION_SYSTEM_INSTRUCTIONS,
    GENERATE_INSPECTION_PATCH_SYSTEM_INSTRUCTIONS,
    GENERATE_DEBUGGING_REFLECTION_SYSTEM_INSTRUCTIONS,
    GENERATE_PATCH_SYSTEM_INSTRUCTIONS,
)
from src.docker_utils.basic_container import SimpleDockerSandbox

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def find_function_end_line(file_path: str, start_line: int) -> int:
    """
    Find the end line of a function/class starting at start_line using AST.
    """
    try:
        with open(file_path, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        
        best_node = None
        min_distance = float('inf')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Exact match is always preferred
                if node.lineno == start_line:
                    return node.end_lineno
                
                # Check for fuzzy match (decorators or line-offset from tracer)
                dist = abs(node.lineno - start_line)
                if dist <= 5 and dist < min_distance:
                    # Also ensure the start_line is actually within the node's body
                    # or very close to the start
                    if node.lineno <= start_line <= node.end_lineno or dist < 2:
                        best_node = node
                        min_distance = dist
        
        if best_node:
            return best_node.end_lineno
            
    except Exception as e:
        print(f"DEBUG: Error in find_function_end_line: {e}")
    return 0


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
    logger.debug(
        "Built prompt (problem_chars=%s, prompt_chars=%s)", len(problem), len(prompt)
    )
    return prompt


def build_hypothesis_prompt(problem: str) -> str:
    problem = (problem or "").strip()
    prompt = f"""{GENERATE_HYPOTHESIS_SYSTEM_INSTRUCTIONS}

    Problem:
    {problem}

    """
    return prompt


_FENCED_CODE_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)\s*```", re.MULTILINE
)


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


def get_default_llm_connector(
    model_name: str | None = None, temperature: float = 0.5
):
    """
    Returns the default LLM connector.
    Prioritizes Gemini if GOOGLE_API_KEY is set, otherwise defaults to Ollama.
    """
    google_api_key = os.environ.get("GEMINI_API_KEY")

    if google_api_key:
        model = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        llm = GeminiLLMConnector(
            model_name=model, temperature=temperature, api_key=google_api_key
        )
        logger.info(
            "Created Gemini LLM connector (model=%s, temperature=%s)",
            model,
            temperature,
        )
    else:
        model = model_name or os.environ.get("OLLAMA_MODEL", "gemma3:12b")
        llm = OllamaLLMConnector(model_name=model, temperature=temperature)
        logger.info(
            "Created Ollama LLM connector (model=%s, temperature=%s)",
            model,
            temperature,
        )
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


def build_evidence_evaluation_prompt(
    problem: str, hypothesis: str, evidence: str, code: str
) -> str:
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


def get_function_source(node: dict) -> str:
    """Get source code for a call graph node."""
    file_path = node.get("file", "")
    start_line = int(node.get("start_line", 0))
    end_line = int(node.get("end_line", 0))

    # Force print for debugging since logging might be misconfigured
    print(f"DEBUG: get_function_source for {node.get('fqn')} in {file_path} at lines {start_line}-{end_line}")

    if not file_path:
        return ""

    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"DEBUG: File NOT FOUND: {file_path}")
        return ""

    if end_line <= 0:
        end_line = find_function_end_line(file_path, start_line)
        print(f"DEBUG: AST found end_line: {end_line}")

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            
            if start_line <= 0 or start_line > len(lines):
                print(f"DEBUG: Invalid start line {start_line} (file has {len(lines)} lines)")
                return ""

            if end_line > 0 and end_line < start_line:
                print(f"DEBUG: Invalid range detected: start_line {start_line} > end_line {end_line}")
                # Fallback to 30 lines
                extracted = "".join(lines[start_line - 1 : start_line + 29])
            elif end_line <= 0:
                extracted = "".join(lines[start_line - 1 : start_line + 29])
            else:
                extracted = "".join(lines[start_line - 1 : end_line])
            
            if not extracted:
                print(f"DEBUG: Extracted source is empty")
            
            return extracted
    except Exception as e:
        print(f"DEBUG: Exception in get_function_source: {e}")
        return ""


def run_in_sandbox(patch: str) -> str:
    """Run code in a sandbox and return the output."""
    with SimpleDockerSandbox() as sandbox:
        # Create the inspection script
        sandbox.run_command("cat << 'EOF' > inspection.py\n" + patch + "\nEOF")
        # Run it
        exit_code, stdout, stderr = sandbox.run_command("python3 inspection.py")
        result = f"Exit Code: {exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        return result


def apply_function_source(node: dict, new_source: str) -> bool:
    """Replace a function's source code in its file."""
    file_path = node.get("file", "")
    start_line = node.get("start_line", 0)
    end_line = node.get("end_line", 0)

    if not file_path or not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return False
    
    if end_line <= 0:
        end_line = find_function_end_line(file_path, start_line)

    logger.info(f"Applying patch to {file_path} at lines {start_line}-{end_line}")
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        if start_line <= 0:
            logger.warning(f"Invalid start line for node {node.get('fqn')}")
            return False

        # If end_line is still 0, we can't reliably replace the function.
        if end_line <= 0:
            logger.warning(f"Could not determine end line for node {node.get('fqn')}")
            return False

        # new_source should be a string of the full function body
        # Ensure it ends with a newline
        if not new_source.endswith("\n"):
            new_source += "\n"

        # Replace the lines
        prefix = lines[: start_line - 1]
        suffix = lines[end_line:]
        new_lines = prefix + [new_source] + suffix

        with open(file_path, "w") as f:
            f.writelines(new_lines)

        logger.info(f"Successfully applied patch to {file_path} at lines {start_line}-{end_line}")
        return True
    except Exception as e:
        logger.error(f"Error applying patch to {file_path}: {e}")
        return False


def run_command(command: str, container_id: str | None = None, workdir: str | None = None) -> str:
    """Run a command either locally or in a Docker container."""
    if container_id:
        logger.info(f"Running command in container {container_id}: {command}")
        try:
            client = docker.from_env()
            container = client.containers.get(container_id)
            
            # Use bash -c to support pipes, redirects, etc.
            exec_res = container.exec_run(
                cmd=["bash", "-c", command],
                workdir=workdir,
                demux=True
            )
            
            stdout = exec_res.output[0].decode("utf-8", errors="replace") if exec_res.output[0] else ""
            stderr = exec_res.output[1].decode("utf-8", errors="replace") if exec_res.output[1] else ""
            
            return (
                f"Exit Code: {exec_res.exit_code}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )
        except Exception as e:
            return f"Error running command in container: {e}"
    else:
        return run_local_command(command)


def run_local_command(command: str) -> str:
    """Run a command locally and return the output."""
    logger.info(f"Running local command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )
        return (
            f"Exit Code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def build_inspection_patch_prompt(target_node: str, source_code: str) -> str:
    # Use simple replacement to interpolate the target node into the instructions
    system_instructions = GENERATE_INSPECTION_PATCH_SYSTEM_INSTRUCTIONS.replace("{target_node}", target_node)
    prompt = f"""{system_instructions}

    Target Function: {target_node}
    Source Code:
    {source_code}
    """
    return prompt


def build_debugging_reflection_prompt(
    target_node: str, source_code: str, patch: str, execution_result: str
) -> str:
    prompt = f"""{GENERATE_DEBUGGING_REFLECTION_SYSTEM_INSTRUCTIONS}

    Target Function: {target_node}
    Source Code:
    {source_code}

    Inspection Patch:
    {patch}

    Execution Result:
    {execution_result}
    """
    return prompt


def build_patch_prompt(target_node: str, source_code: str, reflection: str) -> str:
    prompt = f"""{GENERATE_PATCH_SYSTEM_INSTRUCTIONS}

    Target Function: {target_node}
    Source Code:
    {source_code}

    Reflection:
    {reflection}
    """
    return prompt


def find_test_files_for_node(call_graph: dict, target_fqn: str) -> list[str]:
    """
    Find files that call the target node and look like test files.
    """
    test_files = set()
    
    # 1. Get all callers
    callers = [
        edge["source"] 
        for edge in call_graph.get("edges", []) 
        if edge["target"] == target_fqn
    ]
    
    # 2. Map callers to files and filter for tests
    for caller_fqn in callers:
        caller_node = next(
            (n for n in call_graph.get("nodes", []) if n["fqn"] == caller_fqn), 
            None
        )
        if caller_node:
            file_path = caller_node.get("file", "")
            filename = os.path.basename(file_path)
            if filename.startswith("test_") or "tests/" in file_path or "/tests" in file_path:
                test_files.add(file_path)
                
    return list(test_files)


def save_history(history: list[dict], file_path: str = "artifacts/agent_history.json") -> None:
    """Save the agent's history to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved agent history to {file_path}")
    except Exception as e:
        logger.error(f"Error saving history to {file_path}: {e}")


def generate_diff(original: str, patched: str, filename: str = "source.py") -> str:
    """
    Generate a unified diff between original and patched source code.
    """
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}"
    )
    return "".join(diff)
