"""
decision_engine.py

Combines:
  - the structured claim (from claim_extractor)
  - the grounded image findings (from image_analyzer)
  - the user's risk history (from user_history.csv)
into the final decision fields: claim_status, risk_flags, severity,
claim_status_justification.

This is deliberately the ONLY place user_history influences the output.
The brief is explicit that history should "add risk context" but never
"override clear visual evidence by itself" — so history can only ADD a
risk flag and push borderline cases toward manual_review_required, it
can never flip claim_status from supported to contradicted on its own,
and it can never change issue_type/object_part/severity, which stay
fully image-grounded. Keeping that rule in one function (not scattered
across the pipeline) makes it auditable.

No LLM call is needed here — this step is pure rule application over
already-extracted structured fields, so it's free and instant. That's
also why it doesn't appear in evaluation_report.md's per-call cost
table: there's nothing to meter.
"""

from schema import RISK_FLAGS, SEVERITY


def decide(
    image_findings: dict,
    claim_object: str,
    user_history_row: dict | None,
    explicit_scope_note: str | None,
    multiple_issues_mentioned: bool,
) -> dict:
    """
    Returns the final decision dict with keys:
    evidence_standard_met, evidence_standard_met_reason, risk_flags,
    issue_type, object_part, claim_status, claim_status_justification,
    supporting_image_ids, valid_image, severity
    """
    risk_flags = set(image_findings.get("image_quality_risk_flags", []))

    # valid_image: usable for automated review at all. This is about
    # whether we could form a decision, not about whether the claim
    # turned out to be supported.
    image_load_errors = image_findings.get("image_load_errors", [])
    has_any_supporting_images = bool(image_findings.get("supporting_image_ids"))
    valid_image = image_findings.get("claim_vs_image_match") != "insufficient" or has_any_supporting_images
    if not image_findings.get("per_image") and not has_any_supporting_images:
        valid_image = False

    evidence_standard_met = bool(image_findings.get("evidence_standard_met", False))
    evidence_standard_met_reason = image_findings.get(
        "evidence_standard_met_reason", "Insufficient information to evaluate evidence standard."
    )

    issue_type = image_findings.get("issue_type", "unknown")
    object_part = image_findings.get("object_part", "unknown")
    severity = image_findings.get("severity", "unknown")
    if severity not in SEVERITY:
        severity = "unknown"

    supporting_image_ids = image_findings.get("supporting_image_ids", []) or []

    match = image_findings.get("claim_vs_image_match", "insufficient")
    if not evidence_standard_met or not valid_image:
        claim_status = "not_enough_information"
        justification = evidence_standard_met_reason
    elif match == "matches":
        claim_status = "supported"
        justification = image_findings.get("evidence_standard_met_reason", "Image evidence supports the claim.")
    elif match == "contradicts":
        claim_status = "contradicted"
        justification = "Image evidence does not match what the customer described."
        risk_flags.add("claim_mismatch")
    else:
        claim_status = "not_enough_information"
        justification = "Image evidence is inconclusive for this claim."

    if explicit_scope_note:
        justification += f" Customer scope note: {explicit_scope_note}"

    if image_load_errors:
        risk_flags.add("cropped_or_obstructed")

    # --- user history: additive risk only, never overrides claim_status above ---
    history_risk_triggered = False
    if user_history_row is not None:
        history_risk_triggered = _apply_history_risk(user_history_row, risk_flags)

    needs_manual_review = (
        claim_status == "not_enough_information"
        or "possible_manipulation" in risk_flags
        or "claim_mismatch" in risk_flags
        or history_risk_triggered
    )
    if needs_manual_review:
        risk_flags.add("manual_review_required")

    if multiple_issues_mentioned and len(supporting_image_ids) <= 1:
        # More than one issue claimed but evidence only clearly covers
        # one — flag it rather than silently picking one to report.
        risk_flags.add("wrong_object_part") if object_part == "unknown" else None

    risk_flags = {f for f in risk_flags if f in RISK_FLAGS}
    risk_flags_str = ";".join(sorted(risk_flags)) if risk_flags else "none"

    return {
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": evidence_standard_met_reason,
        "risk_flags": risk_flags_str,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": ";".join(supporting_image_ids) if supporting_image_ids else "none",
        "valid_image": valid_image,
        "severity": severity,
    }


def _apply_history_risk(user_history_row: dict, risk_flags: set) -> bool:
    """
    Mutates risk_flags in place with history-derived flags.
    Returns True if a history-based risk was triggered (used by the
    caller to decide on manual_review_required).
    """
    triggered = False
    try:
        rejected = int(user_history_row.get("rejected_claim", 0) or 0)
        manual_review = int(user_history_row.get("manual_review_claim", 0) or 0)
        recent = int(user_history_row.get("last_90_days_claim_count", 0) or 0)
    except (TypeError, ValueError):
        rejected = manual_review = recent = 0

    history_flags_field = (user_history_row.get("history_flags") or "").strip().lower()
    has_explicit_flag = history_flags_field not in ("", "none")

    if rejected > 0 or has_explicit_flag or recent >= 3:
        risk_flags.add("user_history_risk")
        triggered = True

    return triggered
