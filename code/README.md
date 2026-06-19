# Claims verification pipeline

Verifies damage claims using submitted images, the customer's chat
transcript, user history, and a minimum-evidence rulebook.

## What was broken, and what changed

The original `output.csv` was not a CSV — it was a single Python dict
literal (`output_row = {...}`) that had been written straight to a
`.csv` path, with several fields hardcoded (`evidence_standard_met:
True`, `risk_flags: "none"`, `supporting_image_ids: "img_1"`) instead
of computed per claim. This project replaces that with a real,
four-stage pipeline (`claim_extractor` → `image_analyzer` →
`decision_engine` → `main`) that computes every field from the
specific row's transcript, images, and history, and writes the full
result set in one pass with `csv.DictWriter`.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
```

Place the image folders so that paths in the CSVs resolve correctly,
e.g. if a row says `image_paths: images/test/case_001/img_1.jpg`, that
file should exist at `<project_root>/images/test/case_001/img_1.jpg`
(or pass `--images-root` pointing at wherever your images live).

## Project layout

```
project/
├── dataset/
│   ├── claims.csv
│   ├── sample_claims.csv
│   ├── user_history.csv
│   └── evidence_requirements.csv
├── images/
│   ├── sample/
│   └── test/
├── src/
│   ├── schema.py            # shared allowed-values + config, single source of truth
│   ├── claim_extractor.py   # text-only: parses the chat transcript
│   ├── image_analyzer.py    # vision call: the only module that looks at pixels
│   ├── decision_engine.py   # combines extraction + vision + history, no LLM call
│   └── main.py               # orchestrates rows, writes output.csv correctly
├── evaluation/
│   ├── run_evaluation.py
│   └── evaluation_report.md
└── output.csv
```

## Running

Evaluate against the labeled sample set first (per the brief):

```bash
PYTHONPATH=src python3 evaluation/run_evaluation.py
```

Run on the real, unlabeled test set:

```bash
cd project
PYTHONPATH=src python3 src/main.py \
  --input dataset/claims.csv \
  --output output.csv \
  --history dataset/user_history.csv \
  --requirements dataset/evidence_requirements.csv \
  --images-root .
```

Use `--limit N` to smoke-test on the first N rows before a full run —
useful given API cost/latency (see `evaluation/evaluation_report.md`).

## Design notes

- **One vision call per claim, not per image.** All of a claim's
  images are sent together in a single multi-image message, with the
  model instructed to judge each image individually inside its
  reasoning. This satisfies the brief's "each image considered
  separately" requirement without paying for N separate round trips
  per claim.
- **User history can only add risk, never flip a decision.** History
  is applied exclusively inside `decision_engine.py`, and only ever
  adds a `user_history_risk` / `manual_review_required` flag — it
  cannot change `claim_status` away from what the images showed, per
  the brief's explicit instruction that history "should not override
  clear visual evidence by itself."
- **Failed rows are recorded, never faked.** If a row's pipeline call
  fails after retries, it's logged to `output.errors.json` and
  excluded from `output.csv` rather than filled with a placeholder
  row — so a failure is always visible, never silently invisible in
  the output.
- **Booleans are written as `"true"`/`"false"`** (lowercase strings)
  to match the casing already used in `sample_claims.csv`.
