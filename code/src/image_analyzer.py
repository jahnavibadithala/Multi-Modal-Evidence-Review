"""
image_analyzer.py

This is the only module that looks at pixels. It sends each claim's
images to a vision-capable LLM together with the customer's claimed
issue and the relevant evidence requirement, and asks for a grounded,
per-image and per-claim judgement.

Design choice worth calling out: ALL images for one claim are sent in a
SINGLE vision call, not one call per image. The brief's
evidence_requirements.csv explicitly says "each submitted image should
be considered separately" — but separate API calls would mean the
model can't compare images against each other (e.g. a wide shot for
context plus a close-up for detail), and would multiply cost/latency
for no benefit. One call with multiple images, instructed to evaluate
each image individually inside its reasoning, satisfies the requirement
without the multiplied cost. This tradeoff is logged in the evaluation
report.
"""

import base64
import json
import mimetypes
import os
import time

from anthropic import Anthropic

from schema import ISSUE_TYPE, MAX_RETRIES, OBJECT_PART, RETRY_BASE_DELAY_SECONDS, VISION_MODEL

_client = Anthropic()

VISION_SYSTEM_PROMPT = """You inspect submitted photo evidence for a damage insurance claim.
You will be shown one or more images (each labeled with its image_id), the
customer's claimed issue, the object type, and the minimum evidence standard
required to evaluate this type of claim.

Ground every judgement in what is actually visible. Do not assume the
customer's claim is correct — the images are the primary source of truth.
If the images don't show what's needed to judge the claim, say so.

allowed issue_type values: dent, scratch, crack, glass_shatter, broken_part,
missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown

allowed object_part values depend on claim_object — you will be given the
exact allowed list for this claim's object type.

Respond with ONLY a JSON object, no preamble, no markdown fences:
{
  "per_image": [
    {"image_id": "img_1", "shows_claimed_object": true, "shows_claimed_part": true,
     "quality_issue": "none, blurry, cropped_or_obstructed, low_light_or_glare, or wrong_angle",
     "notes": "one short sentence on what this image actually shows"}
  ],
  "issue_type": "best single value from the allowed list, based on combined images",
  "object_part": "best single value from the allowed list",
  "evidence_standard_met": true or false,
  "evidence_standard_met_reason": "one sentence, grounded in which image(s) do or don't satisfy the standard",
  "supporting_image_ids": ["image_ids that actually support the final decision, empty list if none"],
  "image_quality_risk_flags": ["any of: blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, possible_manipulation, non_original_image, text_instruction_present - empty list if none"],
  "claim_vs_image_match": "matches, contradicts, or insufficient",
  "severity": "none, low, medium, high, or unknown"
}"""


def analyze_images(
    image_paths: list[str],
    claim_object: str,
    claimed_issue_description: str,
    claimed_parts: list[str],
    evidence_requirement_text: str,
    images_root: str,
) -> dict:
    """
    image_paths: relative paths as given in the CSV, e.g.
                 ["images/test/case_001/img_1.jpg", ...]
    images_root: filesystem root that image_paths are relative to
                 (the project root, since CSV paths already include
                 the leading "images/..." segment).

    Returns the parsed JSON dict described in VISION_SYSTEM_PROMPT,
    plus an injected "image_load_errors" list for any file that
    couldn't be read, so a missing/corrupt file never gets silently
    treated as "the model said it's fine."
    """
    content_blocks = []
    image_load_errors = []
    image_ids_sent = []

    for rel_path in image_paths:
        image_id = os.path.splitext(os.path.basename(rel_path))[0]
        full_path = os.path.join(images_root, rel_path)

        try:
            media_type, _ = mimetypes.guess_type(full_path)
            if media_type is None:
                media_type = "image/jpeg"
            with open(full_path, "rb") as f:
                encoded = base64.standard_b64encode(f.read()).decode("utf-8")
        except (FileNotFoundError, OSError) as exc:
            image_load_errors.append({"image_id": image_id, "path": rel_path, "error": str(exc)})
            continue

        content_blocks.append({"type": "text", "text": f"image_id: {image_id}"})
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": encoded},
        })
        image_ids_sent.append(image_id)

    allowed_parts = sorted(OBJECT_PART.get(claim_object, {"unknown"}))

    instruction_text = (
        f"claim_object: {claim_object}\n"
        f"allowed object_part values: {', '.join(allowed_parts)}\n"
        f"customer claims: {claimed_issue_description}\n"
        f"customer-mentioned parts: {', '.join(claimed_parts) if claimed_parts else 'none specified'}\n"
        f"minimum evidence standard for this claim type: {evidence_requirement_text}\n"
    )
    if image_load_errors:
        instruction_text += (
            f"\nNote: {len(image_load_errors)} image(s) could not be loaded and are "
            "NOT included below. Base your judgement only on the images actually shown."
        )

    content_blocks.insert(0, {"type": "text", "text": instruction_text})

    if not content_blocks or not image_ids_sent:
        # No usable images at all — this is a deterministic case, not a
        # model decision, so we short-circuit and never call the API.
        return _no_usable_images_result(image_load_errors)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.messages.create(
                model=VISION_MODEL,
                max_tokens=800,
                system=VISION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content_blocks}],
            )
            raw_text = response.content[0].text.strip()
            raw_text = _strip_json_fences(raw_text)
            parsed = json.loads(raw_text)
            parsed["image_load_errors"] = image_load_errors
            _sanitize_result(parsed, claim_object)
            return parsed
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = exc
            time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(f"image_analyzer failed after {MAX_RETRIES} attempts: {last_error}")


def _no_usable_images_result(image_load_errors: list[dict]) -> dict:
    return {
        "per_image": [],
        "issue_type": "unknown",
        "object_part": "unknown",
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "No usable images could be loaded for this claim.",
        "supporting_image_ids": [],
        "image_quality_risk_flags": [],
        "claim_vs_image_match": "insufficient",
        "severity": "unknown",
        "image_load_errors": image_load_errors,
    }


def _strip_json_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


def _sanitize_result(parsed: dict, claim_object: str) -> None:
    """
    Defensive normalization: if the model returns a value outside the
    allowed set (it sometimes will), fall back to 'unknown' rather than
    writing an invalid value to output.csv. This is the check that
    would have caught the hardcoded-True bug pattern if it had snuck
    in here instead of decision_engine.py.
    """
    if parsed.get("issue_type") not in ISSUE_TYPE:
        parsed["issue_type"] = "unknown"
    allowed_parts = OBJECT_PART.get(claim_object, {"unknown"})
    if parsed.get("object_part") not in allowed_parts:
        parsed["object_part"] = "unknown"
