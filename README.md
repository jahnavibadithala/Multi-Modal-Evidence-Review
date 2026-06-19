Project/
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
