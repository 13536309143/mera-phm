# Bounded real API validation

Three causal strata were selected before execution: a visible curve, a missing current curve with valid history, and a pre-repair historical image. The same saved forecaster was used for deterministic control, one-context single Agent, and independent-context multi Agent. API calls were uncached. The provider/model is recorded in `protocol.json`.

| Mode | Planned cases | Completed cases | Shared-budget-censored cases | API requests | Measured tokens |
|---|---:|---:|---:|---:|---:|
| Deterministic control | 3 | 3 | 0 | 0 | 0 |
| Single Agent | 3 | 2 | 1 | 10 | 15,104 |
| Multi Agent | 3 | 1 | 2 | 27 | 33,066 |

Total measured usage: **48,170 tokens**, within the nominal 50,000-token experiment allocation. Admission used a conservative UTF-8 byte proxy for prompt size plus reserved completion tokens; it was not an exact provider tokenizer or financial cap. Three API mode-runs were censored when the shared budget could not admit another request. The execution order and shared allocation affect coverage: do not compare the raw fractions above as architecture completion-rate estimates.

One matched API pair completed for the visible-curve stratum. Its single Agent used 6,612 tokens in four requests; multi Agent used 17,210 in twelve. This is a descriptive integration observation, not a statistically established cost multiplier or reliability claim.

Canonical prediction and interval fields remained consistent with the fixed numerical forecaster in all nine records, including censored runs. That property follows from the system's architecture and is not a regression improvement caused by Agent dialogue. Six generated numeric explanations were rejected in the multi-Agent runs. Natural-language semantic accuracy was **not** established and remains explicitly unverified.

`records.json` includes all outcomes, errors, measured usage, censorship flags, context counts and tool-validation events. No credentials, raw inspection text, response request IDs, or private filesystem paths are included. Mock-provider fault tests are separate software checks, not additional real-API observations.
