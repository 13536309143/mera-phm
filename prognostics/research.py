"""Reproducible held-out research, routing ablations, stress tests and paired group resampling."""

import copy
import json
from pathlib import Path

import numpy as np

from .data import dataset_fingerprint, load_dataset
from .models import EXPERTS, POLICIES, ModelBank, score


def make_split(rows, seed):
    equipment = np.array(sorted({r["equipment"] for r in rows}))
    np.random.default_rng(seed).shuffle(equipment)
    divisions = {
        "train": equipment[:25],
        "routing": equipment[25:30],
        "calibration": equipment[30:40],
        "test": equipment[40:],
    }
    index = {
        part: [i for i, r in enumerate(rows) if r["equipment"] in ids]
        for part, ids in divisions.items()
    }
    return index, {part: ids.tolist() for part, ids in divisions.items()}


def evaluate(bank, rows, y):
    p = bank.expert_batch(rows)
    output = {}
    for policy in list(EXPERTS) + [p for p in POLICIES if p not in EXPERTS]:
        pred = bank.policy_batch(rows, policy, p)
        result = score(y, pred)
        groups = np.array([r["equipment"] for r in rows])
        result["equipment_macro_mae"] = float(
            np.mean(
                [np.mean(np.abs(y[groups == g] - pred[groups == g])) for g in np.unique(groups)]
            )
        )
        if policy in bank.radii:
            radius = bank.radii[policy]["radius"]
            covered = np.abs(y - pred) <= radius
            result.update(
                {
                    "interval_radius": radius,
                    "interval_width": 2 * radius,
                    "row_coverage": float(covered.mean()),
                    "whole_equipment_coverage": float(
                        np.mean([covered[groups == g].all() for g in np.unique(groups)])
                    ),
                }
            )
        result["subgroups"] = {}
        masks = {
            "empty_image": np.array([not r["image"][0] for r in rows]),
            "visible_image": np.array([bool(r["image"][0]) for r in rows]),
            "pre_repair_memory": np.array([bool(r["history"][11]) for r in rows]),
            "long_horizon": np.array([r["numeric"][0] > 25 for r in rows]),
            "target_ge_80": y >= 80,
        }
        for name, mask in masks.items():
            if mask.any():
                result["subgroups"][name] = {
                    "n": int(mask.sum()),
                    "mae": float(np.mean(np.abs(y[mask] - pred[mask]))),
                }
        output[policy] = result
    return output, p


def paired_bootstrap(y, a, b, groups, seed):
    delta = np.array(
        [
            np.mean(np.abs(y[groups == g] - a[groups == g]))
            - np.mean(np.abs(y[groups == g] - b[groups == g]))
            for g in np.unique(groups)
        ]
    )
    samples = (
        np.random.default_rng(seed).choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
    )
    return {
        "macro_mae_improvement": float(delta.mean()),
        "equipment_bootstrap_95_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def stress_rows(rows, kind):
    out = copy.deepcopy(rows)
    for row in out:
        if kind == "image_outage":
            row["image"] = [0.0] * 9
            row["history"][:11] = [0.0] * 11
            row["event_history"][:11] = [0.0] * 11
        elif kind == "text_outage":
            row["text"] = ""
        elif kind == "image_history_stale":
            row["image"] = [0.0] * 9
            row["history"][9] = max(row["history"][9], 100.0)
            row["history"][11] = 1.0
            row["event_history"][:11] = [0.0] * 11
            row["event_history"][11] = 1.0
    return out


def train_research(
    root, out, seed=20261012, trees=100, rows_and_y=None, backend="trees", device="cpu", epochs=80
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, y = load_dataset(root) if rows_and_y is None else rows_and_y
    split, devices = make_split(rows, seed)
    print("Equipment splits:", devices, flush=True)
    bank = ModelBank(seed, trees, backend, device, epochs).fit(rows, y, split)
    bank.manifest = {
        "devices": devices,
        "seed": seed,
        "target": "next scheduled inspection degradation level",
        "pipeline_version": "1.1.0",
        "dataset_fingerprint": dataset_fingerprint(root),
        "data_history": "same dataset previously explored; holdout is an internal research split, not untouched external confirmation",
        "train_equipment_count": 25,
        "routing_equipment_count": 5,
        "calibration_equipment_count": 10,
        "test_equipment_count": 10,
        "backend": backend,
        "requested_device": device,
        "training_device": next(iter(bank.models.values())).training_device
        if backend == "neural"
        else "cpu",
        "epochs": epochs if backend == "neural" else None,
    }
    test = [rows[i] for i in split["test"]]
    ty = y[split["test"]]
    metrics, p = evaluate(bank, test, ty)
    tests = {
        kind: evaluate(bank, stress_rows(test, kind), ty)[0]
        for kind in ["image_outage", "text_outage", "image_history_stale"]
    }
    groups = np.array([r["equipment"] for r in test])
    comparisons = {
        name: paired_bootstrap(
            ty,
            bank.policy_batch(test, baseline, p),
            bank.policy_batch(test, "learned_router", p),
            groups,
            seed,
        )
        for name, baseline in [
            ("static_to_router", "static_fusion"),
            ("event_history_to_router", "event_history"),
        ]
    }
    result = {
        "manifest": bank.manifest,
        "metrics": metrics,
        "stress_tests": tests,
        "paired_comparisons": comparisons,
        "calibration": bank.radii,
        "training_seconds": bank.training_seconds,
        "protocol": "Fixed expert hyperparameters; separate equipment for fitting, routing, calibration and test.",
        "novelty_status": "candidate mechanism, not a proven world-first or publication claim",
        "agent_effect": "Not measured by regression ablation; use agent-control command for orchestration audit.",
    }
    bank.save(out / "model-bank.joblib")
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    policy_predictions = {name: bank.policy_batch(test, name, p) for name in POLICIES}
    predictions = [
        {
            "equipment": r["equipment"],
            "row": r["row"],
            "time": r["time"],
            "truth": float(actual),
            "predictions": {name: float(values[i]) for name, values in policy_predictions.items()},
        }
        for i, (r, actual) in enumerate(zip(test, ty))
    ]
    (out / "held-out-predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 多模态退化预测研究实验",
        "",
        f"种子：{seed}。设备划分：25 训练、5 路由、10 校准、10 测试。",
        "",
        "这是同一模拟数据集的内部研究验证；该数据集此前已被探索，不是全新外部测试。",
        "",
        "| 方法 | MAE adu | RMSE adu | 整台设备区间覆盖率 |",
        "|---|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        coverage = m.get("whole_equipment_coverage")
        report.append(
            f"| {name} | {m['mae']:.4f} | {m['rmse']:.4f} | {'—' if coverage is None else f'{coverage:.1%}'} |"
        )
    improvement = comparisons["static_to_router"]["macro_mae_improvement"]
    report += [
        "",
        f"学习路由相对静态融合的设备平均 MAE 改善为 {improvement:.4f} adu；负值表示退步，不能隐去。",
        "",
        "缺图、文本中断和旧图像失效压力测试见 results.json。上述压力测试下校准交换性可能失效，覆盖率需实测，不能沿用名义保证。",
        "",
        "Agent 的作用需另测工具顺序、结果一致性、标签隔离、失败率和 token 消耗；不能把模型路由改善归于语言模型协作。",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return bank, result


def run_sweep(root, out, seeds, trees=100):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(root)
    summary = []
    for seed in seeds:
        _, result = train_research(root, out / f"seed-{seed}", seed, trees, dataset)
        summary.append(
            {
                "seed": seed,
                "metrics": result["metrics"],
                "comparisons": result["paired_comparisons"],
            }
        )
    aggregate = {
        policy: {
            "mean_mae": float(np.mean([s["metrics"][policy]["mae"] for s in summary])),
            "std_mae": float(np.std([s["metrics"][policy]["mae"] for s in summary], ddof=1))
            if len(summary) > 1
            else None,
        }
        for policy in POLICIES
    }
    (out / "summary.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "runs": summary,
                "aggregate": aggregate,
                "limitations": "Repeated equipment splits overlap; seed runs are not independent new datasets.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return aggregate
