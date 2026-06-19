"""
evaluation/run_evaluation.py

Runs the pipeline against dataset/sample_claims.csv (which has labels)
and compares predictions to the ground-truth columns, per the brief's
requirement to "evaluate your system" before running on claims.csv.

Usage (from project root):
    PYTHONPATH=src python3 evaluation/run_evaluation.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from main import process_row, load_user_history, load_evidence_requirements  # noqa: E402

COMPARE_FIELDS = [
    "evidence_standard_met", "risk_flags", "issue_type", "object_part",
    "claim_status", "supporting_image_ids", "valid_image", "severity",
]


def main():
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "dataset" / "sample_claims.csv"
    history_path = project_root / "dataset" / "user_history.csv"
    requirements_path = project_root / "dataset" / "evidence_requirements.csv"

    user_history = load_user_history(str(history_path))
    requirements = load_evidence_requirements(str(requirements_path))

    with open(sample_path, newline="", encoding="utf-8") as f:
        labeled_rows = list(csv.DictReader(f))

    per_field_correct = {field: 0 for field in COMPARE_FIELDS}
    total = 0
    mismatches = []

    for row in labeled_rows:
        total += 1
        try:
            prediction = process_row(row, user_history, requirements, str(project_root))
        except Exception as exc:  # noqa: BLE001
            mismatches.append({"user_id": row["user_id"], "error": str(exc)})
            continue

        for field in COMPARE_FIELDS:
            expected = row[field].strip().lower()
            actual = str(prediction[field]).strip().lower()
            if expected == actual:
                per_field_correct[field] += 1
            else:
                mismatches.append({
                    "user_id": row["user_id"], "field": field,
                    "expected": expected, "actual": actual,
                })

    print(f"Evaluated {total} labeled rows from sample_claims.csv\n")
    print("Per-field accuracy:")
    for field in COMPARE_FIELDS:
        pct = (per_field_correct[field] / total * 100) if total else 0
        print(f"  {field:30s} {per_field_correct[field]:3d}/{total}  ({pct:.0f}%)")

    print(f"\nTotal mismatches/errors logged: {len(mismatches)}")
    if mismatches:
        print("First few:")
        for m in mismatches[:10]:
            print(" ", m)


if __name__ == "__main__":
    main()
