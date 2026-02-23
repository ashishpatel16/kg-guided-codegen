import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def build_config() -> dict[str, Any]:
    """Build the config dict expected by UnitTestGenerator."""
    config_path: str = os.path.join(
        os.path.dirname(__file__),
        "src", "agent", "test_generation", "config", "config.json",
    )
    with open(config_path, "r") as f:
        raw: dict[str, Any] = json.load(f)

    return {
        "llm": {
            "model_name": raw.get("model_choice", "gemini-2.0-flash"),
            "max_improvements": raw.get("max_improvements", 2),
            "similarity_comparison_count": raw.get("similarity_comparison_count", 10),
            "max_tests": raw.get("max_tests", 25),
        },
        "api": {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "groq_api_key": os.getenv("GROQ_API_KEY", ""),
            "jina_api_key": os.getenv("JINA_API_KEY", ""),
            "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        },
    }


def setup_loggers() -> tuple[logging.Logger, logging.Logger]:
    """Create a detailed and minimal logger pair."""
    detailed: logging.Logger = logging.getLogger("test_gen.detailed")
    minimal: logging.Logger = logging.getLogger("test_gen.minimal")

    for logger in [detailed, minimal]:
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(name)s][%(levelname)s] %(message)s")
    )
    minimal.addHandler(handler)

    return detailed, minimal


DEMO_CODE: str = '''\
def calculate_grade(score: int) -> str:
    """Assign a letter grade based on a numeric score (0-100)."""
    if score < 0 or score > 100:
        raise ValueError(f"Score must be between 0 and 100, got {score}")

    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
'''

DEMO_COVERAGE_MATRIX: str = """\
No existing coverage data — this is the first test being generated.
"""

DEMO_UNCOVERED_LINES: list[dict[str, Any]] = [
    {"line_number": 4, "line": 'raise ValueError(f"Score must be between 0 and 100, got {score}")'},
    {"line_number": 7, "line": 'return "A"'},
    {"line_number": 9, "line": 'return "B"'},
    {"line_number": 11, "line": 'return "C"'},
    {"line_number": 13, "line": 'return "D"'},
    {"line_number": 15, "line": 'return "F"'},
]


def main() -> None:
    dry_run: bool = "--dry-run" in sys.argv

    print("=" * 60)
    print("Test Generation Agent — Smoke Test")
    print("=" * 60)

    cfg: dict[str, Any] = build_config()
    loggers: tuple[logging.Logger, logging.Logger] = setup_loggers()

    print(f"\nModel: {cfg['llm']['model_name']}")
    print(f"Max improvements: {cfg['llm']['max_improvements']}")
    print(f"Similarity comparison count: {cfg['llm']['similarity_comparison_count']}")

    # Import here so import errors are visible with a clear traceback
    from src.agent.test_generation.core.generator import UnitTestGenerator
    from src.agent.test_generation.core.langchain_graph import LangChainGraph, GraphState

    print("\n✅ All imports successful")
    print(f"✅ GraphState keys: {list(GraphState.__annotations__.keys())}")
    print(f"✅ LangChainGraph available: {LangChainGraph}")
    print(f"✅ UnitTestGenerator available: {UnitTestGenerator}")

    if dry_run:
        print("\n🏁 Dry run complete — imports and config verified. Skipping LLM calls.")
        return

    # Full run — needs API key
    try:
        generator: UnitTestGenerator = UnitTestGenerator(cfg=cfg, loggers=loggers)
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM: {e}")
        print("\nHint: Set GOOGLE_API_KEY in your .env file for Gemini models,")
        print("      or change model_choice in config.json to a GPT model and set OPENAI_API_KEY.")
        print("\nYou can run with --dry-run to verify imports without needing API keys.")
        sys.exit(1)

    # Initialize vector store (empty — no existing tests)
    existing_tests: list[str] = generator.initialize_vector_store(existing_tests="")
    print(f"Existing tests loaded: {len(existing_tests)}")

    print("\n" + "-" * 60)
    print("Generating test case for calculate_grade()...")
    print("-" * 60 + "\n")

    result: dict[str, str] = generator.generate_test(
        code_to_test=DEMO_CODE,
        coverage_matrix=DEMO_COVERAGE_MATRIX,
        uncovered_lines=DEMO_UNCOVERED_LINES,
    )

    print("\n" + "=" * 60)
    print("GENERATED TEST CASE:")
    print("=" * 60)
    print(result["generated_test_case"])

    print("\n" + "=" * 60)
    print("COMBINED TEST SCRIPT:")
    print("=" * 60)
    print(result["combined_test_script"])


if __name__ == "__main__":
    main()
