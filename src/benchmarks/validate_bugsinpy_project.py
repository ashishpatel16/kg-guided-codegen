"""
BugsInPy Reproducibility Validator.

For a given project, validates every bug instance by:
  1. Checking out the FIXED version → compile → test (expect PASS)
  2. Checking out the BUGGY version → compile → test (expect FAIL)

Classifies each bug and writes results to a CSV file.

Usage:
    uv run python -m src.benchmarks.validate_bugsinpy_project tqdm
    uv run python -m src.benchmarks.validate_bugsinpy_project youtube-dl --limit 5
"""

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Status constants ────────────────────────────────────────────────────────
STATUS_REPRODUCIBLE: str = "REPRODUCIBLE"
STATUS_FIXED_FAILS: str = "FIXED_FAILS"
STATUS_BUGGY_PASSES: str = "BUGGY_PASSES"
STATUS_BOTH_FAIL: str = "BOTH_FAIL"
STATUS_CHECKOUT_ERROR: str = "CHECKOUT_ERROR"
STATUS_COMPILE_ERROR: str = "COMPILE_ERROR"
STATUS_CRASHED: str = "CRASHED"

# ── CSV column order ────────────────────────────────────────────────────────
CSV_COLUMNS: List[str] = [
    "project",
    "bug_id",
    "status",
    "python_version",
    "buggy_commit",
    "fixed_commit",
    "test_file",
    "fixed_checkout_ok",
    "fixed_compile_ok",
    "fixed_test_exit_code",
    "buggy_checkout_ok",
    "buggy_compile_ok",
    "buggy_test_exit_code",
    "error",
    "duration_s",
]


def discover_bug_ids(project_name: str, bugsinpy_root: str) -> List[str]:
    """Return sorted list of bug ID strings for *project_name*."""
    bugs_dir: Path = Path(bugsinpy_root) / "projects" / project_name / "bugs"
    if not bugs_dir.exists():
        logger.error(f"Bugs directory not found: {bugs_dir}")
        return []
    ids: List[str] = [
        d.name for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()
    ]
    ids.sort(key=int)
    return ids


def parse_bug_info(project_name: str, bug_id: str, bugsinpy_root: str) -> Dict[str, str]:
    """
    Parse datasets/BugsInPy/projects/<project>/bugs/<id>/bug.info
    into a dict with keys: python_version, buggy_commit_id, fixed_commit_id, test_file.
    """
    info_path: Path = (
        Path(bugsinpy_root) / "projects" / project_name / "bugs" / bug_id / "bug.info"
    )
    result: Dict[str, str] = {}
    if not info_path.exists():
        return result
    for line in info_path.read_text().splitlines():
        line = line.strip().replace("\r", "")
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"')
    return result


def _run_checkout_compile_test(
    bspy: BugsInPyDockerSandbox,
    version: int,
) -> Dict[str, Any]:
    """
    Checkout a version (0=buggy, 1=fixed) inside an already-running
    container, compile, and run tests.
    """
    step: Dict[str, Any] = {
        "checkout_ok": False,
        "compile_ok": False,
        "test_exit_code": None,
        "error": None,
    }
    exit_code, _out, err = bspy.checkout(version=version)
    if exit_code != 0:
        step["error"] = f"checkout v{version} failed: {err[:200]}"
        return step
    step["checkout_ok"] = True

    exit_code, _out, err = bspy.compile(verbose=True)
    if exit_code != 0:
        step["error"] = f"compile v{version} failed: {err[:200]}"
        return step
    step["compile_ok"] = True

    exit_code, _out, _err = bspy.test(relevant=True, verbose=True)
    step["test_exit_code"] = exit_code
    return step


def classify_bug(fixed_passed: bool, buggy_failed: bool) -> str:
    """Determine status from the two test outcomes."""
    if fixed_passed and buggy_failed:
        return STATUS_REPRODUCIBLE
    if not fixed_passed and buggy_failed:
        return STATUS_FIXED_FAILS
    if fixed_passed and not buggy_failed:
        return STATUS_BUGGY_PASSES
    return STATUS_BOTH_FAIL


def validate_single_bug(
    project_name: str,
    bug_id: str,
    bugsinpy_root: str,
    experiments_dir: str,
) -> Dict[str, Any]:
    """
    Validate one bug instance in a single container.
    Checks out fixed first, then buggy — reusing the same container.
    """
    start: datetime = datetime.now()
    meta: Dict[str, str] = parse_bug_info(project_name, bug_id, bugsinpy_root)

    row: Dict[str, Any] = {
        "project": project_name,
        "bug_id": bug_id,
        "status": STATUS_CRASHED,
        "python_version": meta.get("python_version", ""),
        "buggy_commit": meta.get("buggy_commit_id", ""),
        "fixed_commit": meta.get("fixed_commit_id", ""),
        "test_file": meta.get("test_file", ""),
        "fixed_checkout_ok": False,
        "fixed_compile_ok": False,
        "fixed_test_exit_code": None,
        "buggy_checkout_ok": False,
        "buggy_compile_ok": False,
        "buggy_test_exit_code": None,
        "error": None,
        "duration_s": 0.0,
    }

    try:
        with BugsInPyDockerSandbox(
            project_name=project_name,
            bug_id=bug_id,
            bugsinpy_root=bugsinpy_root,
            experiments_dir=experiments_dir,
            keep_alive=False,
        ) as bspy:
            # ── Fixed version (v=1) ──────────────────────────────────────
            logger.info(f"[{project_name}#{bug_id}] Testing FIXED version...")
            fixed: Dict[str, Any] = _run_checkout_compile_test(bspy, version=1)
            row["fixed_checkout_ok"] = fixed["checkout_ok"]
            row["fixed_compile_ok"] = fixed["compile_ok"]
            row["fixed_test_exit_code"] = fixed["test_exit_code"]

            if fixed["error"] and not fixed["checkout_ok"]:
                row["status"] = STATUS_CHECKOUT_ERROR
                row["error"] = fixed["error"]
                row["duration_s"] = round((datetime.now() - start).total_seconds(), 2)
                return row

            if fixed["error"] and not fixed["compile_ok"]:
                row["status"] = STATUS_COMPILE_ERROR
                row["error"] = fixed["error"]
                row["duration_s"] = round((datetime.now() - start).total_seconds(), 2)
                return row

            # ── Buggy version (v=0) — same container ────────────────────
            logger.info(f"[{project_name}#{bug_id}] Testing BUGGY version...")
            buggy: Dict[str, Any] = _run_checkout_compile_test(bspy, version=0)
            row["buggy_checkout_ok"] = buggy["checkout_ok"]
            row["buggy_compile_ok"] = buggy["compile_ok"]
            row["buggy_test_exit_code"] = buggy["test_exit_code"]

            if buggy["error"] and not buggy["checkout_ok"]:
                row["status"] = STATUS_CHECKOUT_ERROR
                row["error"] = buggy["error"]
                row["duration_s"] = round((datetime.now() - start).total_seconds(), 2)
                return row

            if buggy["error"] and not buggy["compile_ok"]:
                row["status"] = STATUS_COMPILE_ERROR
                row["error"] = buggy["error"]
                row["duration_s"] = round((datetime.now() - start).total_seconds(), 2)
                return row

            # ── Classify ─────────────────────────────────────────────────
            fixed_passed: bool = fixed["test_exit_code"] == 0
            buggy_failed: bool = buggy["test_exit_code"] != 0
            row["status"] = classify_bug(fixed_passed, buggy_failed)

            errors: List[str] = [e for e in [fixed.get("error"), buggy.get("error")] if e]
            row["error"] = "; ".join(errors) if errors else None

    except Exception as exc:
        row["error"] = str(exc)[:300]

    row["duration_s"] = round((datetime.now() - start).total_seconds(), 2)
    return row


def write_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    """Write (or overwrite) the CSV report."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_project(
    project_name: str,
    bugsinpy_root: str = "datasets/BugsInPy",
    experiments_dir: str = "experiments",
    artifacts_dir: str = "artifacts",
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """
    Validate all bugs for *project_name* and write results to CSV.
    Returns the list of per-bug result dicts.
    """
    bug_ids: List[str] = discover_bug_ids(project_name, bugsinpy_root)
    if not bug_ids:
        logger.error(f"No bugs found for project '{project_name}'.")
        return []

    if limit > 0:
        bug_ids = bug_ids[:limit]

    total: int = len(bug_ids)
    logger.info(f"Validating {total} bugs for project '{project_name}'")

    os.makedirs(artifacts_dir, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path: str = os.path.join(
        artifacts_dir, f"validated_{project_name}_{timestamp}.csv"
    )

    results: List[Dict[str, Any]] = []

    for i, bug_id in enumerate(bug_ids, 1):
        logger.info(f"[{i}/{total}] Validating {project_name} #{bug_id}...")
        row: Dict[str, Any] = validate_single_bug(
            project_name, bug_id, bugsinpy_root, experiments_dir,
        )
        results.append(row)

        # Incremental save after each bug
        write_csv(results, csv_path)
        logger.info(
            f"[{i}/{total}] {project_name} #{bug_id} → {row['status']}  "
            f"({row['duration_s']}s)"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    reproducible: int = sum(1 for r in results if r["status"] == STATUS_REPRODUCIBLE)
    logger.info(f"\nDone. {reproducible}/{total} bugs are REPRODUCIBLE.")
    logger.info(f"CSV saved to: {csv_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate BugsInPy bug reproducibility for a project."
    )
    parser.add_argument("project_name", help="BugsInPy project name (e.g. youtube-dl)")
    parser.add_argument(
        "--bugsinpy-root", default="datasets/BugsInPy",
        help="Path to BugsInPy dataset root",
    )
    parser.add_argument(
        "--experiments-dir", default="experiments",
        help="Working directory for checked-out projects",
    )
    parser.add_argument(
        "--artifacts-dir", default="artifacts",
        help="Directory for output CSV",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only validate the first N bugs (0 = all)",
    )
    args = parser.parse_args()

    validate_project(
        project_name=args.project_name,
        bugsinpy_root=args.bugsinpy_root,
        experiments_dir=args.experiments_dir,
        artifacts_dir=args.artifacts_dir,
        limit=args.limit,
    )
