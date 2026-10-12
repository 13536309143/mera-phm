"""Locked, equipment-grouped validation, matched-label baselines and full feature ablations."""

import copy
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import dataset_fingerprint, load_dataset
from .models import EXPERTS, POLICIES, ModelBank, features, group_radius, score
from .research import paired_bootstrap

STRESSES = [
    "image_outage",
    "text_outage",
    "both_outage",
    "stale_history",
    "random_image_missing_50",
    "block_image_outage",
    "curve_feature_noise",
]
BASELINES = [
    "dummy_mean",
    "ridge_fusion_25",
    "random_forest_fusion_25",
    "random_forest_fusion_30",
    "extra_trees_fusion_30",
    "hist_gradient_numeric_25",
]
ABLATIONS = ["router_without_freshness", "router_without_reset"]


def json_write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fold_split(rows, test_devices, seed):
    ids = np.array(sorted({r["equipment"] for r in rows} - set(test_devices)))
    if len(ids) != 40 or len(set(test_devices)) != 10:
        raise ValueError("Protocol requires 50 devices and 10 test devices.")
    np.random.default_rng(seed).shuffle(ids)
    devices = {
        "train": ids[:25].tolist(),
        "routing": ids[25:30].tolist(),
        "calibration": ids[30:40].tolist(),
        "test": sorted(test_devices),
    }
    indices = {
        key: np.array([i for i, r in enumerate(rows) if r["equipment"] in group])
        for key, group in devices.items()
    }
    return indices, devices


def ablated_rows(rows, variant):
    modified = copy.deepcopy(rows)
    for row in modified:
        if variant == "router_without_freshness":
            for key in ["history", "event_history"]:
                row[key][9] = row[key][12] = 0.0
        elif variant == "router_without_reset":
            row["history"][11] = 0.0
            row["event_history"] = row["history"].copy()
        else:
            raise ValueError("Unknown component ablation.")
    return modified


def rebuild_image_memory(rows):
    """Rebuild from surviving observations; never retain the dropped current image in memory."""
    last = {}
    for row in rows:
        equipment = row["equipment"]
        t = row["time"]
        previous = last.get(equipment, ([0.0] * 9, None))
        if row["image"][0]:
            previous = (row["image"].copy(), t)
        last[equipment] = previous
        image, image_time = previous
        age = 0.0 if image_time is None else t - image_time
        before = float(image_time is not None and image_time < t - row["numeric"][1])
        row["history"] = image + [age, float(image_time is not None), before, row["history"][12]]
        row["event_history"] = row["history"].copy()
        if before:
            row["event_history"][:11] = [0.0] * 11
    return rows


def stressed_rows(rows, kind, seed, training_image_std=None):
    modified = copy.deepcopy(rows)
    rng = np.random.default_rng(seed)
    positions = {}
    lengths = {}
    for row in rows:
        lengths[row["equipment"]] = lengths.get(row["equipment"], 0) + 1
    for row in modified:
        equipment = row["equipment"]
        position = positions.get(equipment, 0)
        positions[equipment] = position + 1
        drop = kind in ["image_outage", "both_outage"]
        if kind == "random_image_missing_50":
            drop = rng.random() < 0.5
        if kind == "block_image_outage":
            drop = 0.4 * lengths[equipment] <= position < 0.7 * lengths[equipment]
        if drop:
            row["image"] = [0.0] * 9
        if kind in ["text_outage", "both_outage"]:
            row["text"] = ""
        if kind == "curve_feature_noise" and row["image"][0]:
            values = np.array(row["image"])
            values[1:] = np.clip(
                values[1:] + rng.normal(0, 0.1 * np.asarray(training_image_std)[1:]), 0, 1
            )
            row["image"] = values.tolist()
    if kind != "stale_history":
        rebuild_image_memory(modified)
    else:
        for row in modified:
            row["image"] = [0.0] * 9
            row["history"][9] = 100.0
            row["history"][11] = 1.0
            row["event_history"][:11] = [0.0] * 11
            row["event_history"][11] = 1.0
    return modified


def fit_baselines(bank, rows, y, split, seed, trees):
    train = split["train"]
    combined = np.concatenate([train, split["routing"]])
    specifications = {
        "dummy_mean": (DummyRegressor(strategy="mean"), train, "numeric"),
        "ridge_fusion_25": (
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            train,
            "static_fusion",
        ),
        "random_forest_fusion_25": (
            RandomForestRegressor(
                n_estimators=trees, min_samples_leaf=3, random_state=seed, n_jobs=2
            ),
            train,
            "static_fusion",
        ),
        "random_forest_fusion_30": (
            RandomForestRegressor(
                n_estimators=trees, min_samples_leaf=3, random_state=seed, n_jobs=2
            ),
            combined,
            "static_fusion",
        ),
        "extra_trees_fusion_30": (
            ExtraTreesRegressor(
                n_estimators=trees, min_samples_leaf=3, random_state=seed, n_jobs=2
            ),
            combined,
            "static_fusion",
        ),
        "hist_gradient_numeric_25": (
            HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
            ),
            train,
            "numeric",
        ),
    }
    fitted = {}
    for name, (model, indices, feature_name) in specifications.items():
        start = time.perf_counter()
        model.fit(features([rows[i] for i in indices], bank.vectorizer, feature_name), y[indices])
        fitted[name] = (model, feature_name)
        print(f"Baseline {name}: {time.perf_counter() - start:.2f}s", flush=True)
    return fitted


def all_predictions(bank, rows, baselines):
    experts = bank.expert_batch(rows)
    predictions = {**experts, **{p: bank.policy_batch(rows, p, experts) for p in POLICIES}}
    for name, (model, feature_name) in baselines.items():
        predictions[name] = model.predict(features(rows, bank.vectorizer, feature_name))
    return predictions


def metric_with_radius(y, p, groups, radius):
    metric = score(y, p)
    metric["equipment_macro_mae"] = float(
        np.mean([np.abs(y[groups == g] - p[groups == g]).mean() for g in np.unique(groups)])
    )
    covered = np.abs(y - p) <= radius
    metric.update(
        {
            "row_coverage": float(covered.mean()),
            "whole_equipment_coverage": float(
                np.mean([covered[groups == g].all() for g in np.unique(groups)])
            ),
            "mean_interval_width": float(np.mean(2 * radius)),
        }
    )
    return metric


def run_benchmark(root, out, seed=20261012, trees=100, backend="trees", device="cpu", epochs=80):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, y = load_dataset(root)
    groups = np.array([r["equipment"] for r in rows])
    source = Path(__file__).parent
    neural_runtime = None
    if backend == "neural":
        from .neural import resolve_device, torch_runtime

        neural_runtime = {
            "torch": torch_runtime().__version__,
            "training_device": resolve_device(device),
        }
    protocol = {
        "name": "MERA-PHM five-fold equipment validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "folds": 5,
        "devices": 50,
        "samples": len(rows),
        "expert_trees": trees,
        "backend": backend,
        "neural_runtime": neural_runtime,
        "requested_device": device,
        "neural_epochs": epochs if backend == "neural" else None,
        "neural_architecture": "standardized feature MLP 64/32/GELU, AdamW lr=0.001, batch=512; CPU error router"
        if backend == "neural"
        else None,
        "per_fold_devices": {"training": 25, "routing": 5, "calibration": 10, "test": 10},
        "baseline_label_budget": "25-device and 30-device controls; all share vocabulary fitted on 25 training devices.",
        "baseline_hyperparameters": {
            "min_leaf": 3,
            "ridge_alpha": 10.0,
            "hist_gradient_iterations": 150,
            "hist_gradient_l2": 1.0,
        },
        "component_ablations": {
            "router_without_freshness": "Retrain all experts and gates with image age and text-change age zeroed in both history representations.",
            "router_without_reset": "Retrain all experts and gates without history invalidation or explicit pre-repair flag; raw numeric repair time remains.",
        },
        "stress_conditions": STRESSES,
        "noise_definition": "Descriptor perturbation = 10% of training-feature standard deviation; not raw sensor/image corruption.",
        "missingness_definition": "Additional random masking probability 50%; contiguous image masking from 40% to 70% of each trajectory, rebuilding causal memory.",
        "dataset_fingerprint": dataset_fingerprint(root),
        "python": platform.python_version(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
        },
        "limitations": [
            "Same synthetic dataset previously explored; internal validation, not untouched external confirmation.",
            "Hyperparameters fixed before this run; no original-paper CNN reproduction.",
            "Bootstrap over devices is descriptive; models share training data across folds.",
            "Shifted stress tests do not inherit nominal conformal coverage guarantees.",
        ],
    }
    # Persist the protocol before evaluating any folds.
    json_write(out / "protocol.json", protocol)
    with ZipFile(out / "numerical-source-snapshot.zip", "w", ZIP_DEFLATED) as snapshot:
        for name in ["data.py", "models.py", "research.py", "benchmark.py", "neural.py"]:
            snapshot.write(source / name, "prognostics/" + name)
    methods = list(dict.fromkeys(EXPERTS + POLICIES + BASELINES + ABLATIONS))
    oof = {name: np.zeros(len(rows)) for name in methods}
    radii = {name: np.zeros(len(rows)) for name in methods}
    stress_methods = [
        "numeric",
        "numeric_text",
        "numeric_image",
        "static_fusion",
        "history_fusion",
        "missing_rule",
        "learned_router",
        "router_unconstrained",
        "extra_trees_fusion_30",
        "random_forest_fusion_30",
    ]
    stress_predictions = {
        kind: {name: np.zeros(len(rows)) for name in stress_methods} for kind in STRESSES
    }
    records = []
    assigned = np.zeros(len(rows), dtype=int)
    start = time.perf_counter()
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, test_indices) in enumerate(cv.split(np.zeros(len(rows)), y, groups), 1):
        test_devices = sorted(set(groups[test_indices].tolist()))
        split, devices = fold_split(rows, test_devices, seed + fold)
        for a in devices:
            for b in devices:
                if a != b:
                    assert not set(devices[a]) & set(devices[b])
        print(f"Fold {fold}/5: test devices {test_devices}", flush=True)
        bank = ModelBank(seed + fold, trees, backend, device, epochs).fit(rows, y, split)
        baseline = fit_baselines(bank, rows, y, split, seed + fold, trees)
        test = [rows[i] for i in split["test"]]
        cal = [rows[i] for i in split["calibration"]]
        predictions = all_predictions(bank, test, baseline)
        cp = all_predictions(bank, cal, baseline)
        cg = groups[split["calibration"]]
        ty = y[split["test"]]
        tg = groups[split["test"]]
        fold_metrics = {}
        for name, p in predictions.items():
            radius, _ = group_radius(y[split["calibration"]], cp[name], cg)
            oof[name][split["test"]] = p
            radii[name][split["test"]] = radius
            fold_metrics[name] = metric_with_radius(ty, p, tg, np.full(len(test), radius))
        for variant in ABLATIONS:
            altered = ablated_rows(rows, variant)
            ablation = ModelBank(seed + fold, trees, backend, device, epochs).fit(altered, y, split)
            atest = [altered[i] for i in split["test"]]
            p = ablation.policy_batch(atest, "learned_router")
            radius = ablation.radii["learned_router"]["radius"]
            oof[variant][split["test"]] = p
            radii[variant][split["test"]] = radius
            fold_metrics[variant] = metric_with_radius(ty, p, tg, np.full(len(test), radius))
        std = np.std(np.array([rows[i]["image"] for i in split["train"]]), axis=0)
        stress_info = {}
        for kind in STRESSES:
            modified = stressed_rows(test, kind, seed + fold, std)
            p = all_predictions(bank, modified, baseline)
            for name in stress_methods:
                stress_predictions[kind][name][split["test"]] = p[name]
            stress_info[kind] = {
                "current_image_missing_fraction": float(
                    np.mean([not r["image"][0] for r in modified])
                ),
                "metrics": {
                    name: metric_with_radius(ty, p[name], tg, radii[name][split["test"]])
                    for name in stress_methods
                },
            }
        assigned[split["test"]] += 1
        record = {"fold": fold, "devices": devices, "metrics": fold_metrics, "stress": stress_info}
        records.append(record)
        json_write(out / f"fold-{fold}.json", record)
    assert np.all(assigned == 1), "Every observation must be tested exactly once."
    metrics = {name: metric_with_radius(y, p, groups, radii[name]) for name, p in oof.items()}
    for name, m in metrics.items():
        m["fold_mae_mean"] = float(np.mean([r["metrics"][name]["mae"] for r in records]))
        m["fold_mae_sd"] = float(np.std([r["metrics"][name]["mae"] for r in records], ddof=1))
    comparisons = {
        name: paired_bootstrap(y, p, oof["learned_router"], groups, seed)
        for name, p in oof.items()
        if name != "learned_router"
    }
    stress = {
        kind: {
            name: metric_with_radius(y, p, groups, radii[name]) for name, p in predictions.items()
        }
        for kind, predictions in stress_predictions.items()
    }
    result = {
        "protocol": protocol,
        "metrics": metrics,
        "comparisons_to_router": comparisons,
        "stress": stress,
        "every_device_tested_once": True,
        "seconds": round(time.perf_counter() - start, 2),
    }
    json_write(out / "summary.json", result)
    # Raw targets/predictions remain in ignored local artifacts, never in published results.
    private = Path.cwd() / "artifacts"
    private.mkdir(exist_ok=True)
    np.savez_compressed(
        private / f"benchmark-{backend}-{seed}-{out.name}-oof.npz", truth=y, equipment=groups, **oof
    )
    report = [
        "# MERA-PHM equipment-grouped validation",
        "",
        "Exploratory internal evaluation on an already explored synthetic dataset. Each of the 50 devices is tested once.",
        "",
        "| Method | OOF MAE | Device macro MAE | RMSE | Mean interval width | Whole-device coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        report.append(
            f"| {name} | {m['mae']:.4f} | {m['equipment_macro_mae']:.4f} | {m['rmse']:.4f} | {m['mean_interval_width']:.2f} | {m['whole_equipment_coverage']:.0%} |"
        )
    report += [
        "",
        "The router fits experts on 25 devices and its gate on 5 more. The 30-device static baselines control for this additional label budget.",
        "",
        "## Stress tests",
        "",
        "| Condition | Static fusion MAE | Unconstrained router MAE | Constrained router MAE |",
        "|---|---:|---:|---:|",
    ]
    for kind, m in stress.items():
        report.append(
            f"| {kind} | {m['static_fusion']['mae']:.4f} | {m['router_unconstrained']['mae']:.4f} | {m['learned_router']['mae']:.4f} |"
        )
    report += ["", "## Interpretation limits", ""] + [
        f"- {limitation}" for limitation in protocol["limitations"]
    ]
    report += [
        "",
        "Complete fold assignments, parameters, component definitions, widths, and descriptive paired-device bootstrap estimates are in the adjacent JSON files.",
        "",
    ]
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    return result
