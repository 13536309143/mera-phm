# MERA-PHM research design and evidence

MERA-PHM investigates a specific question: **when observations are missing or describe pre-maintenance equipment states, can evidence validity and learned error routing improve next-inspection prognostics?** Its secondary question is whether bounded independent Agent contexts provide useful, auditable orchestration without corrupting supervised numerical outputs.

This is an implemented research prototype with internal experimental evidence. It is not a confirmed novel invention, a neural reproduction of the dataset paper, or an externally validated maintenance controller.

## Task and representation

Given observations through time t and a known inspection horizon Δ, predict degradation at t+Δ. Degradation is in adu, time in atu. The dataset is synthetic; the numerical metadata concern steam-generator maintenance. Original data and target semantics are documented in [dataset notes](dataset.md).

Four numerical variables, a fixed-frame nine-dimensional curve descriptor, and training-only TF-IDF features feed supervised experts. Image plots are not photographs. The plotting template is the dataset's predetermined blank `1/Image/1.tiff`; it is not fitted to target labels. This extraction assumes identical plotting frames and requires redesign for new image sources.

The online loader does not open `nextinslevel.txt`. Numeric files are read and sliced before feature processing; image/text histories stop at the requested row. Tests establish that unseen future rows do not affect current features, not that low-level file I/O never reads a file's remaining bytes. A globally fixed plotting template is treated as preprocessing metadata, not a live state observation.

## Mechanisms

### Causal evidence memory

Track the latest surviving image and its elapsed age. Infer the latest repair time from current time minus elapsed time since repair. If the historical image predates that repair, invalidate its event-aware image memory. This is a field-semantics-dependent inference, not an explicit repair event log.

Text-change age measures time since content last changed. It is a proxy: unchanged wording can also occur in a genuinely new observation. Age zero does not imply that repair just happened.

### Availability-constrained learned routing

Eight ExtraTrees experts model numeric, text, image, numeric/text, numeric/image, static fusion, ordinary history and maintenance-aware history. A shallow gate, fitted on separate equipment, estimates absolute errors of three candidates: numeric/text, static fusion and event history.

Selection minimizes estimated error over eligible candidates. Static fusion needs a current visible curve; event history needs a valid post-maintenance historical curve. Without either, use numeric/text. Missing text triggers numeric/image if a current curve exists, otherwise numeric. This is hard expert selection, not a trainable end-to-end neural mixture or LLM-generated regression.

The supervisor's choice of language specialists does not change the bank's expert computations or the canonical route. Specialist dialogue is not an input to the predictive gate.

### Equipment-group intervals

On each of ten held-out calibration equipment trajectories, compute its maximum absolute residual. A nominal 90% split-conformal target uses order ceil((10+1)×0.9)=10, so the radius is the largest calibration-group score. Return prediction ± radius without arbitrary clipping.

The intended coverage unit is an entire new equipment trajectory, under exchangeable group trajectories with comparable sampling mechanisms. It does not require independence between rows within equipment. Limited calibration equipment produces broad intervals. Stress-induced shifts do not inherit the nominal guarantee. See the general framework in [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

### Restricted Agent orchestration

Multi Agent provides a supervisor and eight available specialist responsibilities, each with independent messages and a tool allowlist. Require quality inspection, two predictive specialists, fusion, then audit. Single Agent has one context and corresponding evidence/fusion/audit requirements. Deterministic control has no LLM context.

Tools return supervised predictions, causal quality features, and completed training-equipment references. There are no arbitrary-file, shell, key, physical-control or current-query-target tools. Numeric outputs are never parsed from prose. The digit guard rejects Unicode decimal digits but cannot detect every spelled-out number or prove semantic correctness. Original qualitative prose remains unverified; the default report uses canonical fields and a code-generated factual summary.

## Neural backend and computation

The optional PyTorch backend replaces each supervised ExtraTrees expert with a standardized feature MLP: input → 64/GELU → 32/GELU → scalar output. Input and target normalization are fitted on expert-training devices only. Training uses AdamW, learning rate 0.001, weight decay 0.001, batch size 512 and 80 fixed epochs. No test- or calibration-label early stopping is used. The error gate remains an ExtraTrees regressor trained on the independent routing group.

CPU, CUDA and MPS are explicit runtime choices; `auto` prefers CUDA, then MPS, then CPU. Checkpoints carry NumPy CPU weights and restore an inference network on the available runtime. Features and the grouped calibration procedure are unchanged between backends. The curve branch uses descriptors and the text branch uses TF-IDF; the neural backend is a feature MLP comparison.

The added five-fold neural study uses the same partitions and controls as the tree study. MLP static-fusion MAE is 7.1451, routed MAE is 6.6297, versus tree routing 2.6919. It does not establish a neural accuracy advantage; tree models remain the default. The executed study source and its environment are recorded under `experiments/numerical/neural-groupcv`.

## Locked numerical evaluation

Five shuffled GroupKFold partitions test every equipment exactly once. Each fold holds out 10 test devices; the other 40 are randomly assigned to 25 expert training, 5 gate training and 10 calibration. Vocabulary is fitted only on the 25 training devices. Fixed settings are persisted before evaluating folds; no new hyperparameter search is hidden inside this run.

Compare 21 methods: eight experts, gate-input ablations and routing policies, six classical baselines, and two retrained whole-feature ablations. Thirty-device RandomForest and ExtraTrees controls use the same total label-device count available to the routed system. All baselines share the 25-device vocabulary, as disclosed in the protocol.

The **whole-feature freshness ablation** retrains experts and gates after zeroing image/text-change ages in both history representations. The **maintenance-reset ablation** retains raw numerical repair time but removes historical-image invalidation and the explicit pre-repair flag. These differ from the narrower gate-only input ablations; they must not be conflated.

Seven stress conditions include complete image outage, complete text outage, both, stale history, additional random image masking, contiguous image outage and descriptor perturbation. Random and block masks rebuild memory from surviving past images. Perturbation uses training-feature standard deviation, not test distribution statistics, and is not raw pixel/sensor corruption.

[Aggregate results](../experiments/numerical/tree-groupcv/README.md), [parameters and fingerprint](../experiments/numerical/tree-groupcv/protocol.json), each fold, and the original numerical execution-source snapshot are published. Raw labels and per-row predictions stay in ignored local artifacts. Repeated equipment-layer bootstrap estimates are descriptive: models share training sets, and this dataset had already been explored.

## Findings and negative results

- OOF MAE: constrained routing 2.6919, static fusion on 25 devices 2.9759, strong static ExtraTrees on 30 devices 2.7590 adu.
- Relative MAE improvement is about 9.5% over the 25-device static baseline, but about 2.4% over the matched-label static control. The latter paired-device improvement interval includes zero.
- Full freshness removal increases OOF MAE to 2.7406; removing reset increases it to 2.7384. Effects are modest; the reset-effect interval includes zero.
- Whole-image outage shows a clear fallback benefit: static MAE 17.155 versus router 3.998. This supports specific stress resilience, not universal robustness.
- Additional random image masking produces router MAE 6.257, worse than the numeric/text expert's 3.998. The strong static baseline also wins under descriptor perturbation. Shifted historical information can still mislead the gate.
- Router interval width averages 64.15 adu, with 94% observed whole-equipment coverage. The width prevents interpreting coverage as precise decision support.

Earlier repeated-split experiments are preserved separately under `experiments/archive/numerical-pilot`; their better figures are not substituted for the release headline. Eligibility rules were changed after early stress-test inspection. All reported results remain exploratory internal validation on the same synthetic dataset, not pristine external confirmation.

## Agent study and software verification

The expanded opt-in study uses **24 paired cases in six causal quality strata**, four distinct held-out equipment per stratum. It measured **822,665 tokens over 658 requests**, including independent language screening, separate rescue runs and eight specialist smoke tests. No runs were truncated by the shared experiment allocation; individual session and tool-loop limits still apply.

| Mode | Completed / attempted | Requests | Measured tokens |
|---|---:|---:|---:|
| Deterministic local control | 24 / 24 | 0 | 0 |
| Single Agent, default budget | 22 / 24 | 117 | 208,038 |
| Multi-Agent, default budget | 11 / 24 | 272 | 307,492 |
| Multi-Agent rescue on failed cases | 9 / 13 | 191 | 221,522 |

The rescue column is a **fresh, separate run with up to 20 requests**, not a replacement for the default-budget failure. Default sessions permit 12 requests; both settings retain the same 24,000 completion-token ceiling and bounded tool loops. All eight specialist roles completed independent evidence-tool smoke tests. All 85 structured case outputs, including failed sessions and rescues, preserved the fixed model's point and interval outputs.

Multi-Agent orchestration completed less reliably and consumed more tokens than the single-Agent control in this sample. Independent model-assisted screening flagged possible evidence contradictions in 14/22 screened single-Agent runs and 20/24 screened default multi-Agent runs. **These are screening flags, not adjudicated error rates or measured semantic accuracy.** Digit rejection prevents invented numeric prose but does not prove qualitative factual correctness. This release is a research prototype; reliable industrial Agent interpretation is not established.

![Real API completion and cost](assets/agent-validation.png)

See the [expanded protocol and complete ledger](../experiments/agents/paired-api/README.md). The earlier [48,170-token budget-censored pilot](../experiments/archive/agent-pilot/README.md) remains separate and is not pooled with this study.

Offline tests use original temporary fixtures and scripted provider responses to cover causal invariance, calibration/group boundaries, dropped-image memory leakage, unauthorized tools, invalid sequencing, digit rejection, provider error sanitization, unknown usage and budgets. They are implementation verification, not extra real LLM robustness trials. Fresh-install demo execution checks packaging and dashboard assets.

## Relation to existing work and remaining evidence

[Yang, Baraldi and Zio](https://doi.org/10.1016/j.jmsy.2021.01.007) already study numerical/image/text prognostics. [PHM Copilots](https://papers.phmsociety.org/index.php/phmconf/article/view/3906) and [PHMForge](https://arxiv.org/abs/2604.01532) already study algorithm-tool-grounded PHM agents. [QA-MoE](https://www.ijcai.org/proceedings/2026/766) studies quality-aware multimodal expert selection in a different domain. Therefore multimodal fusion, more Agent roles, retrieval, quality gating and tool invocation are not standalone originality claims.

The candidate contribution is their precise maintenance/evidence-validity formulation and auditable protocol in this task. Stronger evidence would include a prespecified untouched dataset or independently generated domain trajectories, comparable multimodal neural baselines, explicit maintenance-event annotations, uncertainty-width improvements, and prespecified Agent trials with independent human semantic adjudication. A novelty or patentability determination has not been made. No claim of guaranteed publication, external plant accuracy or semantic Agent correctness is supported.
