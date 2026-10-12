"""Opt-in real API comparison. Published output contains aggregates, not credentials or raw texts."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

from .agents import AgentSession
from .cli import load_bank
from .data import discover_data, observations


class ExperimentBudget:
    def __init__(self, total_tokens=50000, max_requests=60):
        self.total_tokens = total_tokens
        self.max_requests = max_requests
        self.used = 0
        self.requests = 0
        self.unknown_usage = False

    def admit(self, messages, tools, limit):
        if self.unknown_usage or self.requests >= self.max_requests:
            raise ValueError("Experiment API budget exhausted or usage unknown.")
        # Conservative admission proxy, not a provider-specific tokenizer or monetary cap.
        prompt_bound = (
            len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode())
            + 256
        )
        limit = min(limit, self.total_tokens - self.used - prompt_bound)
        if limit < 512:
            raise ValueError("Insufficient remaining experiment budget for another request.")
        self.requests += 1
        return limit

    def record(self, response):
        if response.usage is None:
            self.unknown_usage = True
        else:
            self.used += response.usage.total_tokens


def selected_cases(bank, root):
    rows = [
        r for equipment in bank.manifest["devices"]["test"] for r in observations(root, equipment)
    ]
    definitions = [
        ("visible_curve", lambda r: bool(r["image"][0])),
        (
            "missing_curve_valid_history",
            lambda r: not r["image"][0] and bool(r["event_history"][10]),
        ),
        ("pre_repair_history", lambda r: bool(r["history"][11])),
    ]
    return [(name, next(r for r in rows if predicate(r))) for name, predicate in definitions]


def run_agent_benchmark(root, out, model=None, env_file=None, max_tokens=50000):
    if env_file:
        for key, value in dotenv_values(env_file).items():
            if value and key.startswith(("MODEL_", "API_")):
                os.environ[key] = value
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bank, identity = load_bank(model)
    budget = ExperimentBudget(max_tokens)
    cases = selected_cases(bank, root)
    plan = {
        "cases": [
            {"stratum": name, "equipment": row["equipment"], "row": row["row"]}
            for name, row in cases
        ],
        "modes": ["deterministic_control", "single_agent", "multi_agent"],
        "model_sha256": identity,
        "api_model": os.getenv("MODEL_NAME", ""),
        "approximate_total_token_budget": max_tokens,
        "selection": "Causal quality strata on model-held-out devices; no future labels used.",
        "limitations": "Small exploratory integration sample; numeric predictor fixed across modes. Semantic language accuracy not established.",
    }
    (out / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    records = []
    for stratum, row in cases:
        reference = bank.forecast(row)
        for mode in ["deterministic_control", "single_agent", "multi_agent"]:
            session = AgentSession(bank, row, api_budget=budget)
            start = time.perf_counter()
            if mode == "deterministic_control":
                result = session.deterministic()
            elif mode == "single_agent":
                result = session.single_agent()
            else:
                result = session.api()
            record = {
                "stratum": stratum,
                "equipment": row["equipment"],
                "row": row["row"],
                "mode": mode,
                "process_completed": result["agent_completion"],
                "error": result["error"],
                "prediction_consistent": bool(
                    np.isclose(
                        reference["prediction_adu"],
                        result["forecast"]["prediction_adu"],
                        rtol=0,
                        atol=1e-10,
                    )
                ),
                "interval_consistent": bool(
                    np.allclose(
                        reference["interval_adu"],
                        result["forecast"]["interval_adu"],
                        rtol=0,
                        atol=1e-10,
                    )
                ),
                "requests": result["usage"]["requests"],
                "tokens": result["usage"]["total_tokens"],
                "usage_complete": result["usage"]["usage_complete"],
                "seconds": round(time.perf_counter() - start, 3),
                "independent_contexts": result["independent_llm_contexts"],
                "completed_evidence_roles": result["completed_specialists"],
                "validation_rejections": sum(
                    x["event"] == "validation_rejected" for x in result["trace"]
                ),
                "tool_errors": sum("tool_error" in x.get("result", {}) for x in result["trace"]),
                "language_semantically_verified": False,
            }
            record["budget_censored"] = bool(
                record["error"]
                and (
                    "experiment budget" in record["error"].lower()
                    or "Experiment API budget exhausted" in record["error"]
                )
            )
            records.append(record)
            print(
                stratum, mode, record["process_completed"], record["tokens"], "tokens", flush=True
            )
            (out / "records.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n"
            )
    aggregates = {}
    for mode in plan["modes"]:
        part = [r for r in records if r["mode"] == mode]
        aggregates[mode] = {
            "cases": len(part),
            "completed": sum(r["process_completed"] for r in part),
            "numeric_consistent": sum(
                r["prediction_consistent"] and r["interval_consistent"] for r in part
            ),
            "requests": sum(r["requests"] for r in part),
            "tokens": sum(r["tokens"] for r in part),
            "mean_seconds": float(np.mean([r["seconds"] for r in part])),
            "language_validation_rejections": sum(r["validation_rejections"] for r in part),
        }
        aggregates[mode]["budget_censored_cases"] = sum(r["budget_censored"] for r in part)
        aggregates[mode]["uncensored_cases"] = len(part) - aggregates[mode]["budget_censored_cases"]
    summary = {
        "protocol": plan,
        "aggregate": aggregates,
        "measured_api_tokens": budget.used,
        "usage_unknown": budget.unknown_usage,
        "requests_attempted": budget.requests,
        "language_semantic_accuracy": "not measured; process completion is not semantic correctness",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--env-file")
    parser.add_argument("--output", default="results/agent-validation")
    parser.add_argument("--max-tokens", type=int, default=50000)
    args = parser.parse_args()
    run_agent_benchmark(
        discover_data(args.data), args.output, args.model, args.env_file, args.max_tokens
    )


if __name__ == "__main__":
    main()
