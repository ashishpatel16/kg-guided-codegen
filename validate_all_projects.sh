#!/bin/bash
# Quick smoke test: validate 3 bugs from each BugsInPy project.
# If 1-2 compile/pass → project is likely viable.
# If all 3 fail → project is probably broken.

set -e

PROJECTS=(
  youtube-dl
)

echo "============================================="
echo "  BugsInPy Project Smoke Test (3 bugs each)"
echo "============================================="
echo ""

for project in "${PROJECTS[@]}"; do
  echo ">>> Starting: $project"
  uv run python -m src.benchmarks.validate_bugsinpy_project "$project" --limit 45
  echo ""
  echo ">>> Finished: $project"
  echo "---------------------------------------------"
  echo ""
done

echo "============================================="
echo "  All projects done. Check artifacts/ for CSVs."
echo "============================================="
