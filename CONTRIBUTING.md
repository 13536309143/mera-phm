# Contributing to MERA-PHM

Use Python 3.12 or later. Install the project in editable mode and the development tools:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check prognostics tests scripts main.py
ruff format --check prognostics tests scripts main.py
```

Tests generate their own temporary data and mock provider responses. They require neither private API credentials nor the paper dataset.

Keep changes focused. For a new predictive mechanism, describe the hypothesis, its input availability at inference time, the exact component removed in ablations, and its training-label budget. Keep equipment groups disjoint across expert fitting, routing, calibration and testing. Never choose a mechanism using an allegedly untouched test set and continue calling that set untouched.

Publish measured results, including negative results and interval widths. Separate numerical prediction performance from LLM orchestration performance. Keep failed and budget-censored API runs in the evaluation ledger; do not count censored runs as successful or intrinsic agent failures. Mock-provider tests must remain clearly distinguished from real API validation.

Keep credentials, trained model binaries, private inspection records, local paths and private run histories out of commits. The original paper dataset is versioned only through `data/raw/Multimodal data for prognostics.zip`, its manifest and attribution notice. Do not commit extracted dataset copies or generated local artifacts. If the source bundle changes, update its checksum and verify its identity against the intended experiment dataset. Generated toy fixtures are software checks, not industrial accuracy evidence.

Use pull requests for changes. Include relevant verification and identify any claim that existing experiments do not establish.
