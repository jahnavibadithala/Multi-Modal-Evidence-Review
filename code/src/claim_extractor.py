"""
claim_extractor.py

Turns the messy chat transcript in `user_claim` into a structured,
predicted claim: what the customer says is wrong, with which part,
and how many distinct issues they mentioned.

This step never looks at images — it's pure text understanding, so it
uses a cheap model. Keeping it separate from image_analyzer.py matters
for the debugging workflow this project is built around: if the final
output blames the wrong object_part, you want to know whether the
customer's *words* were misread, or the *image* was misread. Splitting
the steps gives you that boundary for free.
"""

import json
import time

from anthropic import Anthropic

from schema import EXTRACTOR_MODEL, MAX_RETRIES, RETRY_BASE_DELAY_SECONDS

_client = Anthropic()

EXTRACTION_SYSTEM_PROMPT = """You read a short customer support chat transcript about a damage claim.
Extract ONLY what the customer claims happened — do not guess at things they didn't say.

Respond with ONLY a JSON object, no preamble, no markdown fences:
{
  "claimed_parts": ["short phrase per distinct part/area mentioned, e.g. 'rear bumper'"],
  "claimed_issue_description": "one sentence summary of what the customer says is wrong",
  "explicit_scope_note": "any explicit instruction from the customer about what to include/exclude, or null",
  "multiple_issues_mentioned": true or false
}"""


def extract_claim(user_claim_text: str, claim_object: str) -> dict:
    """
    Calls the LLM once on the transcript text. Returns a dict with the
    structured fields above. Raises on repeated failure rather than
    silently returning a placeholder — a caller that wants a fallback
    should catch this explicitly, so failures are never invisible.
    """
    user_prompt = (
        f"claim_object: {claim_object}\n\n"
        f"transcript:\n{user_claim_text}"
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.messages.create(
                model=EXTRACTOR_MODEL,
                max_tokens=400,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text.strip()
            raw_text = _strip_json_fences(raw_text)
            parsed = json.loads(raw_text)
            _validate_extraction_shape(parsed)
            return parsed
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = exc
            time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - surfaced via last_error
            last_error = exc
            time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(
        f"claim_extractor failed after {MAX_RETRIES} attempts: {last_error}"
    )


def _strip_json_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


def _validate_extraction_shape(parsed: dict) -> None:
    required_keys = {
        "claimed_parts",
        "claimed_issue_description",
        "explicit_scope_note",
        "multiple_issues_mentioned",
    }
    missing = required_keys - parsed.keys()
    if missing:
        raise KeyError(f"extraction response missing keys: {missing}")
