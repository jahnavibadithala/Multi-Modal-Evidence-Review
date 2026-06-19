# Evaluation report

## Accuracy on sample_claims.csv

Run `PYTHONPATH=src python3 evaluation/run_evaluation.py` and paste the
per-field accuracy table here once run against the real API and real
images. The script compares every predicted field against the labeled
columns in `dataset/sample_claims.csv` and reports per-field accuracy
plus a list of mismatches for manual review.

_Fill in after a real run:_

| Field | Accuracy |
|---|---|
| evidence_standard_met | |
| risk_flags | |
| issue_type | |
| object_part | |
| claim_status | |
| supporting_image_ids | |
| valid_image | |
| severity | |

## Operational analysis

### Model calls per claim
- 1 text-only call (`claim_extractor`) — cheap model
- 1 vision call (`image_analyzer`) — all images for that claim sent
  together in a single call, not one call per image
- 0 calls for `decision_engine` — pure rule logic over already
  extracted/observed fields, no LLM needed

So: **2 model calls per claim**, regardless of image count.

- sample_claims.csv: 20 rows → ~40 calls
- claims.csv: 44 rows → ~88 calls

### Token usage (approximate, fill in real numbers after a run)
- Extraction call: short transcript in, small JSON out — roughly
  150–400 input tokens, 100–150 output tokens per call.
- Vision call: each image costs roughly 1,000–1,600 tokens depending
  on resolution (per Anthropic's image token guidance), plus the
  instruction text (~150–250 tokens) and JSON output (~200–300
  tokens). A claim with 2 images is roughly 2,500–3,500 input tokens.

_Fill in actual totals from `response.usage.input_tokens` /
`output_tokens`, summed across the real run — both modules already
return the raw API response object before parsing, so this is a one-line
addition to log._

### Images processed
- sample_claims.csv: count of all `image_paths` entries across 20 rows
- claims.csv: count of all `image_paths` entries across 44 rows

(Fill in exact counts — `wc` the semicolon-split paths, or log a
counter inside `image_analyzer.analyze_images`.)

### Approximate cost (fill in with your actual model's published pricing)
Cost = (extraction input + output tokens) + (vision input + output
tokens), summed per claim, multiplied by per-token pricing for the
model actually used. Since this pipeline uses 2 fixed calls per claim
regardless of image count, cost scales roughly linearly with claim
count, with a per-claim premium for claims with more images.

### Latency / runtime
- `main.py` processes rows sequentially. For N claims at ~2 calls each
  and typical multi-second latency per vision call, expect roughly
  N × (extraction latency + vision latency) total runtime — e.g. 44
  claims at ~3–6s per claim end-to-end is roughly 2–4 minutes
  sequential.
- The pipeline currently runs single-threaded/sequential by design —
  see "considered but not implemented" below for the batching path
  that would reduce this.

### Rate limits, batching, caching, retries
- **Retries**: both LLM-calling modules (`claim_extractor`,
  `image_analyzer`) retry up to `MAX_RETRIES` (3) times with linear
  backoff on a parse failure or transient API error, rather than
  failing the whole run on one bad response.
- **Failure isolation**: a row that exhausts retries is logged to
  `output.errors.json` and skipped, not retried indefinitely and not
  silently faked — this keeps one bad claim from blocking the batch.
- **TPM/RPM**: sequential processing keeps concurrent request volume
  at 1, which is the simplest way to stay under rate limits at this
  scale (tens of rows), at the cost of total runtime.
- **Caching**: not implemented. The most valuable cache target would
  be the `evidence_requirements.csv` lookup result per
  `(claim_object, issue_family)` pair, since it's static across the
  whole run — currently recomputed per row, which is cheap (no LLM
  call) so it wasn't worth the complexity at this scale.
- **Considered but not implemented**: concurrent processing (e.g. a
  small thread/async pool of 3-5 workers) would meaningfully cut
  runtime for the 44-row test set without meaningfully risking rate
  limits at this volume. Left out to keep the submission simple and
  the retry/error-isolation logic easy to reason about; flagged here
  as the first thing to add if throughput became a real constraint.
