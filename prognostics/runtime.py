"""Configuration checks and a single public-experiment catalog."""

import importlib.util
import json
import os
import platform
from urllib.parse import urlparse

from dotenv import dotenv_values

from .data import PROJECT

EXPERIMENTS = {
    "tree_groupcv": "experiments/numerical/tree-groupcv/summary.json",
    "neural_groupcv": "experiments/numerical/neural-groupcv/summary.json",
    "paired_api": "experiments/agents/paired-api/summary.json",
}


def doctor_report(device="auto"):
    config = {**dotenv_values(PROJECT / ".env"), **os.environ}
    key, model = config.get("MODEL_API_KEY", ""), config.get("MODEL_NAME", "")
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "api_configured": bool(
            key and key != "your-api-key" and model and model != "your-tool-calling-model"
        ),
        "api_host": urlparse(config.get("MODEL_BASE_URL", "") or "").hostname,
        "torch_installed": bool(importlib.util.find_spec("torch")),
        "tree_device": "cpu",
        "neural_device": None,
        "device_check_passed": device in {"auto", "cpu"},
    }
    if result["torch_installed"]:
        from .neural import resolve_device, torch_runtime

        torch = torch_runtime()
        result.update(
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            mps_available=bool(torch.backends.mps.is_available()),
        )
        try:
            result["neural_device"] = resolve_device(device)
            result["device_check_passed"] = True
        except ValueError as exc:
            result["device_error"] = str(exc)
    elif device in {"cuda", "mps"}:
        result["device_error"] = "Install PyTorch with the appropriate platform build first."
    return result


def experiment_index():
    result = {}
    for name, location in EXPERIMENTS.items():
        path = PROJECT / location
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        result[name] = {"summary": location}
        if "metrics" in data:
            result[name].update(
                backend=data["protocol"].get("backend", "trees"),
                router_mae=data["metrics"]["learned_router"]["mae"],
                static_fusion_mae=data["metrics"]["static_fusion"]["mae"],
            )
        else:
            result[name].update(api_tokens=data["measured_api_tokens"], modes=data["aggregate"])
    return result
