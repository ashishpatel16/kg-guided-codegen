from datasets import load_dataset


def load_swebench(split="test"):
    """
    Load the SWE-bench dataset.

    Args:
        split (str): The dataset split to load ('train', 'test', etc.)

    Returns:
        Dataset: SWE-bench dataset
    """
    return load_dataset("princeton-nlp/SWE-bench", split=split)


def load_humaneval(split="test"):
    """
    Load the HumanEval dataset.

    Args:
        split (str): The dataset split to load ('test' is usually the only split)

    Returns:
        Dataset: HumanEval dataset
    """
    return load_dataset("openai_humaneval", split=split)


if __name__ == "__main__":
    # Example usage
    swe = load_swebench()
    # humaneval = load_humaneval()

    import json

    with open("artifacts/swebench_ex1.json", "w") as f:
        json.dump(swe[0], f, indent=4)
    # print(f"HumanEval dataset: {humaneval}")

    # Example using Triage class
    # try:
    #     from triage import Triage

    #     # Create a triage instance from the first SWE-bench example
    #     triage = Triage(swe[2])
    #     print(f"Created triage for: {triage}")

    #     triage.setup_repo(work_root="experiments")
    #     print(f"Done setting up repo")

    #     triage.setup_environment()
    #     print(f"Done setting up environment")

    #     triage.apply_patch()
    #     print(f"Done applying patch")

    #     # Uncomment to actually set up the repo and environment
    #     # repo_dir = triage.setup_repo(work_root="./workspaces")
    #     # venv_dir = triage.setup_environment()
    #     # print(f"Repository: {repo_dir}")
    #     # print(f"Virtual environment: {venv_dir}")
    #     # triage.cleanup()  # Clean up when done
    # except ImportError:
    #     print("Could not import Triage class")
