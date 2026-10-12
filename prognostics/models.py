"""Persisted modality experts, freshness/event routing and equipment-group calibration."""

import math
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .data import quality

EXPERTS = [
    "numeric",
    "text",
    "image",
    "numeric_text",
    "numeric_image",
    "static_fusion",
    "history_fusion",
    "event_history",
]
CANDIDATES = ["numeric_text", "static_fusion", "event_history"]
POLICIES = [
    "static_fusion",
    "history_fusion",
    "event_history",
    "missing_rule",
    "learned_router",
    "router_no_age",
    "router_no_event",
    "router_unconstrained",
]


def array(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def features(rows, vectorizer, name):
    n, im = array(rows, "numeric"), array(rows, "image")
    if name == "numeric":
        return n
    if name == "image":
        return im
    tx = vectorizer.transform([r["text"] for r in rows]).toarray().astype(np.float32)
    if name == "text":
        return tx
    if name == "numeric_text":
        return np.column_stack([n, tx])
    if name == "numeric_image":
        return np.column_stack([n, im])
    if name == "static_fusion":
        return np.column_stack([n, im, tx])
    history = array(rows, "history" if name == "history_fusion" else "event_history")
    return np.column_stack([n, im, history, tx])


def router_features(rows, predictions, variant="full"):
    n, im, h = array(rows, "numeric"), array(rows, "image"), array(rows, "history")
    p = np.column_stack([predictions[name] for name in CANDIDATES])
    x = np.column_stack(
        [n, im[:, 0], h[:, 9], h[:, 11], h[:, 12], np.ptp(p, axis=1), p.std(axis=1)]
    )
    if variant == "no_age":
        x[:, [5, 7]] = 0
    if variant == "no_event":
        x[:, [1, 2, 3, 6]] = 0
    return x


def score(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "mre_percent": float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-9)) * 100),
    }


def group_radius(y, p, groups, coverage=0.9):
    maxima = np.array([np.max(np.abs(y[groups == g] - p[groups == g])) for g in np.unique(groups)])
    rank = math.ceil((len(maxima) + 1) * coverage)
    if rank > len(maxima):
        raise ValueError("校准设备数不足以提供所请求覆盖率的有限设备组区间。")
    return float(np.sort(maxima)[rank - 1]), maxima.tolist()


class ModelBank:
    def __init__(self, seed=20261012, trees=100, backend="trees", device="cpu", epochs=80):
        if backend not in {"trees", "neural"}:
            raise ValueError("backend 需为 trees 或 neural。")
        if backend == "trees" and device not in {"cpu", "auto"}:
            raise ValueError("树模型使用 CPU；GPU 训练请添加 --backend neural。")
        self.seed, self.trees = seed, trees
        self.backend, self.device, self.epochs = backend, device, epochs
        self.models = {}
        self.routers = {}
        self.radii = {}
        self.manifest = {}

    def fit(self, rows, y, split):
        train_idx = np.asarray(split["train"])
        route_idx = np.asarray(split["routing"])
        cal_idx = np.asarray(split["calibration"])
        train = [rows[i] for i in train_idx]
        validation = [rows[i] for i in route_idx]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=256)
        self.vectorizer.fit([r["text"] for r in train])
        self.training_seconds = {}
        for name in EXPERTS:
            start = time.perf_counter()
            if self.backend == "neural":
                from .neural import NeuralRegressor

                model = NeuralRegressor(seed=self.seed, epochs=self.epochs, device=self.device)
            else:
                model = ExtraTreesRegressor(
                    n_estimators=self.trees, min_samples_leaf=3, random_state=self.seed, n_jobs=2
                )
            model.fit(features(train, self.vectorizer, name), y[train_idx])
            self.models[name] = model
            self.training_seconds[name] = time.perf_counter() - start
            print("Trained expert:", name, flush=True)
        vp = self.expert_batch(validation)
        errors = np.column_stack([np.abs(y[route_idx] - vp[name]) for name in CANDIDATES])
        for variant in ["full", "no_age", "no_event"]:
            gate = ExtraTreesRegressor(
                n_estimators=80, max_depth=4, min_samples_leaf=30, random_state=self.seed, n_jobs=2
            )
            gate.fit(router_features(validation, vp, variant), errors)
            self.routers[variant] = gate
        calibrate = [rows[i] for i in cal_idx]
        cp = self.expert_batch(calibrate)
        cg = np.array([r["equipment"] for r in calibrate])
        for policy in POLICIES:
            p = self.policy_batch(calibrate, policy, cp)
            radius, maxima = group_radius(y[cal_idx], p, cg)
            self.radii[policy] = {
                "radius": radius,
                "group_max_residuals": maxima,
                "target_coverage": 0.9,
                "calibration_equipment": sorted(set(cg.tolist())),
            }
        self.scaler = StandardScaler().fit(features(train, self.vectorizer, "numeric_image"))
        self.reference_x = self.scaler.transform(features(train, self.vectorizer, "numeric_image"))
        self.references = [
            {
                "equipment": r["equipment"],
                "row": r["row"],
                "time": r["time"],
                "historical_outcome_adu": float(y[i]),
            }
            for r, i in zip(train, train_idx)
        ]
        return self

    def expert_batch(self, rows):
        return {
            name: model.predict(features(rows, self.vectorizer, name))
            for name, model in self.models.items()
        }

    def policy_batch(self, rows, policy, predictions=None):
        p = predictions if predictions is not None else self.expert_batch(rows)
        if policy in EXPERTS:
            return p[policy]
        if policy == "missing_rule":
            return np.where(array(rows, "image")[:, 0] > 0, p["static_fusion"], p["numeric_text"])
        variants = {
            "learned_router": "full",
            "router_no_age": "no_age",
            "router_no_event": "no_event",
            "router_unconstrained": "full",
        }
        if policy not in variants:
            raise ValueError("未知预测策略。")
        variant = variants[policy]
        expected = np.maximum(self.routers[variant].predict(router_features(rows, p, variant)), 0)
        eligible = self.eligibility(rows)
        selected = np.argmin(
            expected if policy == "router_unconstrained" else np.where(eligible, expected, np.inf),
            axis=1,
        )
        matrix = np.column_stack([p[name] for name in CANDIDATES])
        result = matrix[np.arange(len(rows)), selected]
        if policy != "router_unconstrained":
            for i, row in enumerate(rows):
                if not row["text"].strip():
                    result[i] = p["numeric_image" if row["image"][0] else "numeric"][i]
        return result

    @staticmethod
    def eligibility(rows):
        return np.asarray([[True, bool(r["image"][0]), bool(r["event_history"][10])] for r in rows])

    def forecast(self, sample, policy="learned_router"):
        p = self.expert_batch([sample])
        value = float(self.policy_batch([sample], policy, p)[0])
        radius = self.radii.get(policy)
        if radius is None:
            raise ValueError("所选策略没有独立校准区间。")
        selected = policy
        expected = None
        if policy.startswith("router_") or policy == "learned_router":
            variant = {
                "learned_router": "full",
                "router_no_age": "no_age",
                "router_no_event": "no_event",
                "router_unconstrained": "full",
            }[policy]
            expected = np.maximum(
                self.routers[variant].predict(router_features([sample], p, variant))[0], 0
            )
            scores = (
                expected
                if policy == "router_unconstrained"
                else np.where(self.eligibility([sample])[0], expected, np.inf)
            )
            selected = CANDIDATES[int(np.argmin(scores))]
            if not sample["text"].strip() and policy != "router_unconstrained":
                selected = "numeric_image" if sample["image"][0] else "numeric"
        elif policy == "missing_rule":
            selected = "static_fusion" if sample["image"][0] else "numeric_text"
        return {
            "prediction_adu": value,
            "policy": policy,
            "selected_expert": selected,
            "expert_predictions": {name: float(values[0]) for name, values in p.items()},
            "router_expected_errors": None
            if expected is None
            else dict(zip(CANDIDATES, expected.tolist())),
            "eligible_candidates": {
                name: bool(flag) for name, flag in zip(CANDIDATES, self.eligibility([sample])[0])
            },
            "interval_adu": [value - radius["radius"], value + radius["radius"]],
            "interval_target_coverage": radius["target_coverage"],
            "interval_method": "split conformal using maximum residual per calibration equipment",
            "interval_scope": "whole-trajectory target under exchangeable equipment; exploratory dataset; not universal physical guarantee",
            "horizon_atu": sample["numeric"][0],
            "quality": quality(sample),
            "model_seed": self.seed,
            "label_used_in_inference": False,
        }

    def retrieve(self, sample, k=3):
        x = self.scaler.transform(features([sample], self.vectorizer, "numeric_image"))[0]
        distances = np.linalg.norm(self.reference_x - x, axis=1)
        eligible = [
            i for i, ref in enumerate(self.references) if ref["equipment"] != sample["equipment"]
        ]
        nearest = sorted(eligible, key=lambda i: distances[i])[:k]
        return [
            {
                **self.references[i],
                "distance": float(distances[i]),
                "source_split": "training archive",
            }
            for i in nearest
        ]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path):
        bank = joblib.load(path)
        if not isinstance(bank, cls):
            raise ValueError("模型文件类型不正确。")
        return bank
