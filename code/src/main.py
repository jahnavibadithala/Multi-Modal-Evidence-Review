"""
main.py

Orchestrates the full pipeline over an input CSV (claims.csv or
sample_claims.csv) and writes output.csv.

THIS IS WHERE THE ORIGINAL BUG LIVED. The broken output.csv you started
with was a single Python dict literal written straight to a .csv path
-- something like:

    with open("output.csv", "w") as f:
        f.write(str(output_row))

...run once, overwriting itself, with several fields hardcoded
(evidence_standard_met=True, risk_flags="none", supporting_image_ids=
"img_1") instead of computed per row. The fix below:

  1. Builds ONE row-dict per claim by actually calling the pipeline
     (claim_extractor -> image_analyzer -> decision_engine), so every
     field is computed from that row's real data.
  2. Collects all row-dicts into a list across the whole input file.
  3. Writes them ALL at once with csv.DictWriter, which handles quoting,
     the header row, and consistent column order automatically -- so
     a dict can never again be dumped as a literal.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from claim_extractor import extract_claim
from decision_engine import decide
from image_analyzer import analyze_images
from schema import OUTPUT_FIELDS


def load_user_history(history_csv_path: str) -> dict:
    history_by_user = {}
    with open(history_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            history_by_user[row["user_id"]] = row
    return history_by_user


def load_evidence_requirements(requirements_csv_path: str) -> list[dict]:
    with open(requirements_csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_evidence_requirement_text(
    requirements: list[dict], claim_object: str, issue_hint: str
) -> str:
    """
    Picks the most specific matching requirement row for this claim.
    Falls back to the general "all" requirement if nothing more
    specific matches -- so every claim_object always gets *some*
    evidence standard, even ones not explicitly covered.
    """
    issue_hint_lower = (issue_hint or "").lower()

    specific_matches = [
        r for r in requirements
        if r["claim_object"] == claim_object
        and any(term.strip() in issue_hint_lower for term in r["applies_to"].split(","))
    ]
    if specific_matches:
        return specific_matches[0]["minimum_image_evidence"]

    object_matches = [r for r in requirements if r["claim_object"] == claim_object]
    if object_matches:
        return object_matches[0]["minimum_image_evidence"]

    general_matches = [r for r in requirements if r["claim_object"] == "all"]
    if general_matches:
        return general_matches[0]["minimum_image_evidence"]

    return "The claimed object and relevant part should be visible clearly enough to inspect."


def process_row(row: dict, user_history: dict, requirements: list[dict], images_root: str) -> dict:
    """
    Runs the full pipeline for a single input row. Returns a dict with
    ALL fourteen OUTPUT_FIELDS populated and correctly typed -- the
    boundary that main.py's writer depends on.
    """
    user_id = row["user_id"]
    claim_object = row["claim_object"]
    image_paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]

    extraction = extract_claim(row["user_claim"], claim_object)

    evidence_text = find_evidence_requirement_text(
        requirements, claim_object, extraction.get("claimed_issue_description", "")
    )

    image_findings = analyze_images(
        image_paths=image_paths,
        claim_object=claim_object,
        claimed_issue_description=extraction.get("claimed_issue_description", ""),
        claimed_parts=extraction.get("claimed_parts", []),
        evidence_requirement_text=evidence_text,
        images_root=images_root,
    )

    decision = decide(
        image_findings=image_findings,
        claim_object=claim_object,
        user_history_row=user_history.get(user_id),
        explicit_scope_note=extraction.get("explicit_scope_note"),
        multiple_issues_mentioned=extraction.get("multiple_issues_mentioned", False),
    )

    output_row = {
        "user_id": user_id,
        "image_paths": row["image_paths"],
        "user_claim": row["user_claim"],
        "claim_object": claim_object,
        **decision,
    }
    # Normalize booleans to lowercase strings to match sample_claims.csv's
    # observed format ("true"/"false"), not Python's "True"/"False".
    output_row["evidence_standard_met"] = str(output_row["evidence_standard_met"]).lower()
    output_row["valid_image"] = str(output_row["valid_image"]).lower()

    return {field: output_row.get(field, "") for field in OUTPUT_FIELDS}


def run(input_csv_path: str, output_csv_path: str, history_csv_path: str,
        requirements_csv_path: str, images_root: str, limit: int | None = None) -> None:
    user_history = load_user_history(history_csv_path)
    requirements = load_evidence_requirements(requirements_csv_path)

    with open(input_csv_path, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))

    if limit is not None:
        input_rows = input_rows[:limit]

    output_rows = []
    errors = []
    start_time = time.time()

    for i, row in enumerate(input_rows, start=1):
        try:
            output_row = process_row(row, user_history, requirements, images_root)
            output_rows.append(output_row)
            print(f"[{i}/{len(input_rows)}] ok  user_id={row['user_id']}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            errors.append({"row_index": i, "user_id": row.get("user_id"), "error": str(exc)})
            print(f"[{i}/{len(input_rows)}] FAILED user_id={row.get('user_id')}: {exc}", file=sys.stderr)
            # A failed row is recorded, never silently dropped or
            # silently filled with a placeholder -- it's also never
            # added to output_rows, so it won't pollute output.csv
            # with fabricated values.

    elapsed = time.time() - start_time

    # The actual fix: build the full list first, then write it ALL at
    # once with DictWriter. Never write a single row's dict repr().
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"\nWrote {len(output_rows)}/{len(input_rows)} rows to {output_csv_path} "
        f"in {elapsed:.1f}s ({len(errors)} failed)",
        file=sys.stderr,
    )

    if errors:
        errors_path = Path(output_csv_path).with_suffix(".errors.json")
        with open(errors_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"Failed-row details written to {errors_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the claims verification pipeline.")
    parser.add_argument("--input", default="dataset/claims.csv", help="Input CSV path")
    parser.add_argument("--output", default="output.csv", help="Output CSV path")
    parser.add_argument("--history", default="dataset/user_history.csv")
    parser.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    parser.add_argument("--images-root", default=".", help="Root dir that image_paths in the CSV are relative to")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (useful for smoke tests)")
    args = parser.parse_args()

    run(
        input_csv_path=args.input,
        output_csv_path=args.output,
        history_csv_path=args.history,
        requirements_csv_path=args.requirements,
        images_root=args.images_root,
        limit=args.limit,
    )
