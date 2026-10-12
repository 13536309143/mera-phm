# Expanded real API validation

The protocol and source snapshot were frozen before API calls. The numerical bank was fixed, and only causal inputs from its held-out devices were selected. Six strata each contain four distinct devices; the same device can occur in different strata, so 24 cases are not 24 independent equipment trajectories. Paired modes alternate execution order by case index, with three concurrent case workers. No caching or future query labels were used.

The expanded opt-in study uses **24 paired cases in six causal quality strata**, four distinct held-out equipment per stratum. It measured **822,665 tokens over 658 requests**, including independent language screening, separate rescue runs and eight specialist smoke tests. No runs were truncated by the shared experiment allocation; individual session and tool-loop limits still apply.

| Mode | Completed / attempted | Requests | Measured tokens |
|---|---:|---:|---:|
| Deterministic local control | 24 / 24 | 0 | 0 |
| Single Agent, default budget | 22 / 24 | 117 | 208,038 |
| Multi-Agent, default budget | 11 / 24 | 272 | 307,492 |
| Multi-Agent rescue on failed cases | 9 / 13 | 191 | 221,522 |

The rescue column is a **fresh, separate run with up to 20 requests**, not a replacement for the default-budget failure. Default sessions permit 12 requests; both settings retain the same 24,000 completion-token ceiling and bounded tool loops. All eight specialist roles completed independent evidence-tool smoke tests. All 85 structured case outputs, including failed sessions and rescues, preserved the fixed model's point and interval outputs.

Multi-Agent orchestration completed less reliably and consumed more tokens than the single-Agent control in this sample. Independent model-assisted screening flagged possible evidence contradictions in 14/22 screened single-Agent runs and 20/24 screened default multi-Agent runs. **These are screening flags, not adjudicated error rates or measured semantic accuracy.** Digit rejection prevents invented numeric prose but does not prove qualitative factual correctness. This release is a research prototype; reliable industrial Agent interpretation is not established.

![Real API completion and cost](../../docs/assets/agent-validation.png)

See the [expanded protocol and complete ledger](summary.json). The earlier [48,170-token budget-censored pilot](../agent-validation/README.md) remains separate and is not pooled with this study.

## Per-stratum default completion

| Stratum | Single Agent | Multi-Agent |
|---|---:|---:|
| missing_curve_valid_history | 4 / 4 | 1 / 4 |
| no_image_history | 4 / 4 | 2 / 4 |
| pre_repair_history | 3 / 4 | 2 / 4 |
| stale_text | 4 / 4 | 3 / 4 |
| visible_long_horizon | 3 / 4 | 2 / 4 |
| visible_short_horizon | 4 / 4 | 1 / 4 |

## Artifacts and accounting

- `protocol.json`: selected equipment/row identifiers, model identity, provider model, bounds and execution-source hashes.
- `records.json`: every local, single, default multi and rescue record; failures are retained.
- `summary.json`: aggregates and all eight specialist smoke-test ledgers.
- `source-snapshot.zip`: exact Agent/study source at execution; later formatting changes do not invalidate this snapshot.

Total measured usage includes 68,903 tokens in 59 independent screening requests and 16,710 tokens in 19 specialist smoke-test requests, in addition to the mode table. All attempted requests have known usage. The aggregate admission ceiling is conservative and is neither a provider tokenizer nor a monetary cap. Failed default sessions remain failed even when a separate rescue completes. Increasing requests does not change bounded supervisor/specialist loop counts or prove that budget caused every failure.

The audit specialist smoke test receives a code-verified fusion prerequisite. It checks the role/tool integration independently; it does not represent an autonomous end-to-end run. The screening model uses a separate context with canonical evidence, accepted specialist prose and the final narrative where present. Two single-Agent cases had no accepted prose and were not screened. No human gold annotations exist; flags must not be interpreted as established hallucination prevalence.

Raw prose, inspection records, tool traces and per-row labels remain in ignored local artifacts. Public records omit those texts and credentials. Numerical consistency establishes software grounding, not language truth or an Agent-caused MAE benefit. API behavior may vary with provider version, sampling and concurrency; these are exploratory integration observations on synthetic equipment data.
