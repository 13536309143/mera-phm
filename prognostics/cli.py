import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .agents import run_prediction
from .data import PROJECT, discover_data, load_dataset, online_sample
from .models import POLICIES, ModelBank
from .research import run_sweep, train_research


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")


def save_prediction(result, out=None):
    out = Path(out) if out else PROJECT / "runs" / f"prediction-{stamp()}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    f = result["forecast"]
    interval = f["interval_adu"]
    report = (
        f"# 多模态下一次巡检退化预测\n\n运行方式：{result['mode']}\n\n"
        f"预测：{f['prediction_adu']:.4f} adu。跨度：{f['horizon_atu']:.0f} atu。\n\n"
        f"设备组校准区间：[{interval[0]:.4f}, {interval[1]:.4f}] adu，名义目标为整条设备轨迹 90%。\n\n"
        f"路由：{f['selected_expert']}。API 新请求：{result['new_api_requests']}。\n\n"
        f"Agent 流程完成：{result['agent_completion']}。\n\n"
        + result.get("verified_summary", "数值来自本地监督预测器。")
        + (f"\n\nAPI 流程未完成：{result['error']}。" if result["error"] else "")
        + "\n\n## Agent 原始文字（未经语义事实核验）\n\n"
        + (result["narrative"] or "无")
        + "\n\n这是模拟数据集的研究输出；区间假设及实际 token 记录见 result.json。\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    return out


def load_bank(path=None):
    model = Path(path) if path else PROJECT / "artifacts/main-model/model-bank.joblib"
    if path is None and not model.is_file():
        model = PROJECT / "artifacts/demo-model/model-bank.joblib"
    if not model.is_file():
        raise ValueError("尚无训练模型，请先运行 train。")
    return ModelBank.load(model), hashlib.sha256(model.read_bytes()).hexdigest()


def agent_control(bank, root, out, api=False, cases=1, model_identity="default"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for device in bank.manifest["devices"]["test"][:cases]:
        sample = online_sample(root, device, 100)
        local = run_prediction(bank, sample, "local", model_identity=model_identity)
        mediated = (
            run_prediction(bank, sample, "api", cache=False, model_identity=model_identity)
            if api
            else None
        )
        results.append(
            {
                "equipment": device,
                "row": 100,
                "local_prediction": local["forecast"]["prediction_adu"],
                "api_prediction": None
                if mediated is None
                else mediated["forecast"]["prediction_adu"],
                "numeric_consistency": None
                if mediated is None
                else bool(
                    local["forecast"]["selected_expert"] == mediated["forecast"]["selected_expert"]
                    and np.isclose(
                        local["forecast"]["prediction_adu"],
                        mediated["forecast"]["prediction_adu"],
                        rtol=0,
                        atol=1e-10,
                    )
                    and np.allclose(
                        local["forecast"]["interval_adu"],
                        mediated["forecast"]["interval_adu"],
                        rtol=0,
                        atol=1e-10,
                    )
                ),
                "api_completed": None if mediated is None else mediated["agent_completion"],
                "api_result": mediated,
                "local_trace": local["trace"],
            }
        )
    result = {
        "cases": results,
        "api_run": api,
        "purpose": "Same fixed predictor: evaluate orchestration, not regression improvements from explanations.",
    }
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    from .runtime import configure_console

    configure_console()
    parser = argparse.ArgumentParser(
        description="MERA-PHM: multimodal degradation prediction and auditable agent tools"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser(
        "doctor", help="Check Python, API configuration and optional GPU devices."
    )
    doctor.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    sub.add_parser("experiments", help="Show the published numerical and Agent experiment index.")
    agent = sub.add_parser("agent-benchmark", help="Run the stratified, paired real API study.")
    agent.add_argument("--data")
    agent.add_argument("--model")
    agent.add_argument("--output", default="artifacts/experiments/paired-api")
    agent.add_argument("--workers", type=int, default=3)
    agent.add_argument("--max-tokens", type=int, default=2000000)
    demo = sub.add_parser("demo", help="Generate original toy data and train a demo model; no API.")
    demo.add_argument("--data", default="data/demo")
    demo.add_argument("--output", default="artifacts/demo-model")
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--backend", choices=["trees", "neural"], default="trees")
    demo.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    demo.add_argument("--epochs", type=int, default=80)
    benchmark = sub.add_parser(
        "benchmark",
        help="Run five equipment folds, matched-label baselines and component ablations.",
    )
    benchmark.add_argument("--data")
    benchmark.add_argument("--output", default="artifacts/experiments/tree-groupcv")
    benchmark.add_argument("--seed", type=int, default=20261012)
    benchmark.add_argument("--backend", choices=["trees", "neural"], default="trees")
    benchmark.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    benchmark.add_argument("--epochs", type=int, default=80)
    for command in ["audit", "train", "research", "predict", "serve", "agent-control"]:
        part = sub.add_parser(command)
        part.add_argument("--data")
        if command in ["predict", "serve", "agent-control"]:
            part.add_argument("--model")
        if command != "serve":
            part.add_argument("--output")
        if command == "train":
            part.add_argument("--seed", type=int, default=20261012)
            part.add_argument("--backend", choices=["trees", "neural"], default="trees")
            part.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
            part.add_argument("--epochs", type=int, default=80)
        if command == "research":
            part.add_argument(
                "--seeds", type=int, nargs="+", default=[20261012, 20261013, 20261014]
            )
        if command == "predict":
            part.add_argument("--equipment", type=int, default=2)
            part.add_argument("--row", type=int, default=100)
            part.add_argument("--mode", choices=["local", "single", "api"], default="local")
            part.add_argument("--policy", choices=POLICIES, default="learned_router")
            part.add_argument("--no-cache", action="store_true")
        if command == "serve":
            part.add_argument("--port", type=int, default=8765)
            part.add_argument("--open", action="store_true")
        if command == "agent-control":
            part.add_argument("--api", action="store_true")
            part.add_argument("--cases", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            from .runtime import doctor_report

            report = doctor_report(args.device)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["device_check_passed"] else 1
        if args.command == "experiments":
            from .runtime import experiment_index

            print(json.dumps(experiment_index(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "agent-benchmark":
            from .full_agent_benchmark import run

            run(
                discover_data(args.data),
                args.output,
                args.model,
                PROJECT / ".env",
                args.max_tokens,
                args.workers,
            )
            return 0
        if args.command == "demo":
            from .demo import generate_demo

            root = generate_demo(args.data, args.seed)
            train_research(
                root,
                args.output,
                args.seed,
                trees=20,
                backend=args.backend,
                device=args.device,
                epochs=args.epochs,
            )
            print("Demo ready. Run predict or serve with the saved model.")
            return 0
        if args.command == "benchmark":
            from .benchmark import run_benchmark

            run_benchmark(
                discover_data(args.data),
                args.output,
                args.seed,
                backend=args.backend,
                device=args.device,
                epochs=args.epochs,
            )
            return 0
        root = discover_data(args.data)
        if args.command == "audit":
            rows, y = load_dataset(root)
            report = {
                "equipment": 50,
                "samples": len(rows),
                "target_range": [float(y.min()), float(y.max())],
                "empty_image_rows": sum(not r["image"][0] for r in rows),
                "numeric_channels": 4,
                "future_labels_separate_from_observations": all("target" not in r for r in rows),
                "data": str(root),
            }
            out = (
                Path(args.output)
                if args.output
                else PROJECT / "artifacts" / f"audit-{stamp()}.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "train":
            out = Path(args.output) if args.output else PROJECT / "artifacts" / f"model-{stamp()}"
            _, result = train_research(
                root, out, args.seed, backend=args.backend, device=args.device, epochs=args.epochs
            )
            print("模型与结果：", out)
            print("学习路由 MAE：", result["metrics"]["learned_router"]["mae"])
        elif args.command == "research":
            if not 1 <= len(args.seeds) <= 5:
                raise ValueError("研究轮次需在 1 至 5 之间。")
            out = (
                Path(args.output) if args.output else PROJECT / "artifacts" / f"research-{stamp()}"
            )
            run_sweep(root, out, args.seeds)
            print("研究结果：", out)
        elif args.command == "predict":
            bank, identity = load_bank(args.model)
            sample = online_sample(root, args.equipment, args.row)
            result = run_prediction(
                bank, sample, args.mode, args.policy, not args.no_cache, identity
            )
            out = save_prediction(result, args.output)
            print(
                "预测：",
                round(result["forecast"]["prediction_adu"], 4),
                "adu；报告：",
                out / "report.md",
            )
            print(
                "Agent 完成：",
                result["agent_completion"],
                "；新 API 请求：",
                result["new_api_requests"],
            )
            if result["error"]:
                print("原因：", result["error"])
                return 1
        elif args.command == "agent-control":
            if not 1 <= args.cases <= 3:
                raise ValueError("单次 Agent 对照限 1 至 3 个设备样本。")
            bank, identity = load_bank(args.model)
            out = (
                Path(args.output)
                if args.output
                else PROJECT / "artifacts" / f"agent-control-{stamp()}"
            )
            agent_control(bank, root, out, args.api, args.cases, identity)
            print("Agent 对照：", out)
        else:
            from .server import serve

            serve(root, args.model, args.port, args.open)
    except (ValueError, OSError) as exc:
        print("无法完成：", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
