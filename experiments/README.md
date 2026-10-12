# 实验总览 / Experiment index

先区分两个问题：数值模型预测是否准确，以及 Agent 是否可靠完成取证流程。训练后端、压力测试和 Agent 开销分别记录，历史试验归档。

| 类别 / Study | 入口 / Location | 协议 / Protocol | 主要结果 / Main result |
|---|---|---|---|
| 树模型 / Tree prediction | [tree-groupcv](numerical/tree-groupcv/README.md) | 50设备、五折、21方法、7压力条件 | 路由 MAE 2.692 |
| 神经网络 / Neural prediction | [neural-groupcv](numerical/neural-groupcv/README.md) | 同样设备划分，80 epochs，CPU MLP | 路由 MAE 6.630 |
| Agent 协作 / Agent orchestration | [paired-api](agents/paired-api/README.md) | 6情境、24配对样本、8专家工具测试 | 单 Agent 22/24；多 Agent 11/24 |
| 历史 / Archive | [archive](archive/README.md) | 早期试验，不合并主结果 | 保留开发过程与预算截断 |

## Reading order

1. Root README: purpose, deployment, API and training commands.
2. This index: choose numerical accuracy or Agent orchestration.
3. Study `README.md`: complete metrics and interpretation.
4. `protocol.json`: device assignments, fixed settings, data identity and execution-source hashes.
5. `fold-*.json` or `records.json`: per-fold or per-case evidence.
6. Source snapshots: the exact implementation used by the reported run.

Both prediction backends use causal modality descriptors and TF-IDF. Neural experts use standardized 64/32/GELU MLPs; preprocessing and the error gate stay on CPU. The neural study used PyTorch 2.14.1 on CPU, with no API calls. Its execution environment is saved separately.

The Agent study uses the fixed v1.0 forecaster and captured Agent source. File organization and software deployment have since been updated; moving these records does not rerun or change the original study. Higher-budget rescue runs remain separate from default-budget failures. Independent language screening supplies review flags, not human-adjudicated semantic accuracy.

The current data is simulated and previously explored; all prediction results are internal validation. The strong-baseline effect interval includes zero, and uncertainty intervals are relatively wide. CUDA/MPS provide computation options; their availability does not imply better prediction accuracy. Model changes should be evaluated under the same grouped protocol.

## Reproduction

Run `python main.py experiments` to inspect the published catalog. Run `benchmark` for predictions and `agent-benchmark` for real API orchestration. Default local outputs use `artifacts/experiments/`, keeping working runs separate from the published studies.
