Multi-Modal Evidence Review
Build a system that verifies damage claims using images, a short claim conversation, user history, and minimum evidence requirements.

Each claim is about one of three object types: car, laptop, package.

Your system must decide whether the submitted images support the user's claim, contradict it, or do not provide enough information.

The images are the primary source of truth. The user conversation defines what needs to be checked. User history can add risk context, but should not override clear visual evidence by itself.



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
