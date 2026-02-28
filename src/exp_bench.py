import os
import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = "/Users/ashish/master-thesis/kg-guided-codegen"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agent.test_generation.core.generator import UnitTestGenerator
from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

def load_env_vars():
    """Load environment variables from .env if it exists."""
    env_path = Path(project_root) / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

def setup_loggers():
    """Set up the two loggers expected by the agent."""
    detailed_logger = logging.getLogger("detailed")
    detailed_logger.setLevel(logging.INFO)
    if not detailed_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        detailed_logger.addHandler(handler)

    minimal_logger = logging.getLogger("minimal")
    minimal_logger.setLevel(logging.INFO)
    if not minimal_logger.handlers:
        minimal_handler = logging.StreamHandler(sys.stdout)
        minimal_handler.setFormatter(logging.Formatter('%(message)s'))
        minimal_logger.addHandler(minimal_handler)

    return detailed_logger, minimal_logger

def run_bug10_experiment(exp_dir="experiments_testgen_bug10"):
    # 0. Load env vars
    load_env_vars()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        return

    # 1. Setup Loggers
    detailed_logger, minimal_logger = setup_loggers()

    # 2. Setup Sandbox
    project = "youtube-dl"
    bug = "10"
    
    print(f"--- Starting Sandbox for {project} Bug {bug} ---")
    with BugsInPyDockerSandbox(project, bug, experiments_dir=exp_dir) as bspy:
        # 3. Checkout and Compile
        print("Checking out buggy version...")
        bspy.checkout(version=0)
        
        print("Compiling project...")
        exit_code, out, err = bspy.compile(verbose=True)
        if exit_code != 0:
            print(f"Compilation failed: {err}")
            return
            
        # 4. Extract Code
        # test_file="test/test_utils.py"
        # source_file="youtube_dl/utils.py"
        test_file_path = f"{bspy.container_project_root}/test/test_utils.py"
        source_file_path = f"{bspy.container_project_root}/youtube_dl/utils.py"
        
        print(f"Reading failing test: {test_file_path}")
        _, failing_test_code, _ = bspy.sandbox.run_command(f"cat {test_file_path}")
        
        print(f"Reading source module: {source_file_path}")
        _, source_code, _ = bspy.sandbox.run_command(f"cat {source_file_path}")

        # 5. Initialize Agent
        cfg = {
            "llm": {
                "model_name": "gemini-2.0-flash",
                "max_improvements": 1,
                "similarity_comparison_count": 5
            },
            "api": {
                "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "groq_api_key": os.getenv("GROQ_API_KEY", ""),
                "jina_api_key": os.getenv("JINA_API_KEY", "")
            }
        }

        print("\n--- Initializing Test Generation Agent ---")
        generator = UnitTestGenerator(cfg, (detailed_logger, minimal_logger))
        
        # Load the failing test file as "existing context" in the vector store
        print("Loading failing test into vector store...")
        generator.initialize_vector_store(existing_tests=failing_test_code)

        # 6. Generate New Test
        print("\n--- Regenerating Test Case for utils.py ---")
        result = generator.generate_test(
            code_to_test=source_code,
            coverage_matrix="{}",
            uncovered_lines=[],
        )

        print("\n" + "="*50)
        print("GENERATED TEST CASE OUTPUT")
        print("="*50)
        print(result["generated_test_case"])

        print("\n" + "="*50)
