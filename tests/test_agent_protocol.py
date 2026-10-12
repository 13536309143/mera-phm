"""Scripted provider responses exercise failures without making API requests."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prognostics.agents import AgentSession
from prognostics.budgets import ExperimentBudget
from prognostics.data import load_dataset, online_sample
from prognostics.demo import generate_demo
from prognostics.models import ModelBank
from prognostics.research import make_split


class Message:
    def __init__(self, content=None, tools=None):
        self.content = content
        self.tool_calls = [
            SimpleNamespace(
                id=f"tool-{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args))
            )
            for i, (name, args) in enumerate(tools or [])
        ] or None

    def model_dump(self, exclude_none=True):
        result = {
            "role": "assistant",
            "content": self.content,
            "provider_metadata": "opaque-preserved-field",
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {"name": t.function.name, "arguments": t.function.arguments},
                }
                for t in self.tool_calls
            ]
        return result


def tool(name, args=None):
    return Message(tools=[(name, args or {})])


class Backend:
    def __init__(self, responses, usage=True):
        self.responses = list(responses)
        self.usage = usage
        self.seen = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.seen.append(copy.deepcopy(kwargs["messages"]))
        message = self.responses.pop(0)
        if isinstance(message, Exception):
            raise message
        return SimpleNamespace(
            id=f"mock-{len(self.seen)}",
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30)
            if self.usage
            else None,
            choices=[
                SimpleNamespace(
                    message=message, finish_reason="tool_calls" if message.tool_calls else "stop"
                )
            ],
        )

    def close(self):
        self.closed = True


def multi_script(bad_numeric=False):
    responses = [
        tool("inspect_quality"),
        Message(
            tools=[
                ("delegate", {"role": "numeric", "question": "数值证据"}),
                ("delegate", {"role": "text", "question": "文本证据"}),
            ]
        ),
        tool("numeric_prediction"),
    ]
    if bad_numeric:
        responses.append(Message("区间为 88 到 115。"))
    responses += [
        Message("数值证据已获取。"),
        tool("text_evidence"),
        Message("文本证据已获取。"),
        tool("run_fusion"),
        tool("verify_forecast"),
        Message("模型与证据流程已完成。"),
    ]
    return responses


class AgentProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = generate_demo(Path(cls.tmp.name) / "data")
        rows, y = load_dataset(root)
        split, devices = make_split(rows, 17)
        cls.bank = ModelBank(seed=17, trees=8).fit(rows, y, split)
        cls.bank.manifest = {"devices": devices}
        cls.sample = online_sample(root, 2, 20)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def execute(self, responses, mode="multi", usage=True, max_requests=12):
        backend = Backend(responses, usage)
        session = AgentSession(self.bank, self.sample, max_requests=max_requests)

        def configure():
            session.client = backend
            session.model = "mock-model"

        with patch.object(session, "configure_api", side_effect=configure):
            result = session.api() if mode == "multi" else session.single_agent()
        self.assertTrue(backend.closed)
        self.assertEqual(result["forecast"], session.forecast)
        return result, backend

    def test_independent_multi_contexts_and_provider_metadata(self):
        result, backend = self.execute(multi_script())
        self.assertTrue(result["agent_completion"])
        self.assertEqual(result["independent_llm_contexts"], 3)
        self.assertTrue(
            any(
                m.get("provider_metadata") == "opaque-preserved-field"
                for call in backend.seen
                for m in call
            )
        )

    def test_single_context_same_tool_requirements(self):
        result, _ = self.execute(
            [
                tool("inspect_quality"),
                Message(tools=[("numeric_prediction", {}), ("text_evidence", {})]),
                tool("run_fusion"),
                tool("verify_forecast"),
                Message("证据流程已完成。"),
            ],
            mode="single",
        )
        self.assertTrue(result["agent_completion"])
        self.assertEqual(result["independent_llm_contexts"], 1)

    def test_unauthorized_tool_rejected_and_recovery(self):
        result, _ = self.execute([tool("read_file", {"path": ".env"})] + multi_script())
        self.assertTrue(result["agent_completion"])
        self.assertIn("tool_error", result["trace"][0]["result"])

    def test_early_fusion_rejected_and_recovery(self):
        result, _ = self.execute([tool("run_fusion")] + multi_script())
        self.assertTrue(result["agent_completion"])
        self.assertIn("tool_error", result["trace"][0]["result"])

    def test_invented_numeric_interval_rejected_then_corrected(self):
        result, _ = self.execute(multi_script(bad_numeric=True))
        self.assertTrue(result["agent_completion"])
        self.assertEqual(sum(x["event"] == "validation_rejected" for x in result["trace"]), 1)
        self.assertNotIn("区间为 88 到 115。", json.dumps(result["trace"], ensure_ascii=False))

    def test_request_limit_marks_failure_and_retains_prediction(self):
        result, _ = self.execute(multi_script(), max_requests=3)
        self.assertFalse(result["agent_completion"])
        self.assertLessEqual(result["usage"]["requests"], 3)

    def test_unknown_usage_exhausts_conservative_completion_budget(self):
        result, _ = self.execute(multi_script(), usage=False)
        self.assertFalse(result["agent_completion"])
        self.assertFalse(result["usage"]["usage_complete"])

    def test_provider_error_never_echoes_sensitive_exception_text(self):
        error = RuntimeError("SENSITIVE_SENTINEL")
        error.status_code = 429
        result, _ = self.execute([error])
        self.assertFalse(result["agent_completion"])
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(result))
        self.assertIn("429", result["error"])

    def test_experiment_budget_blocks_before_request(self):
        budget = ExperimentBudget(total_tokens=100)
        with self.assertRaises(ValueError):
            budget.admit([], [], 4096)
        self.assertEqual(budget.requests, 0)

    def test_provider_failure_stops_experiment_with_unknown_usage(self):
        error = RuntimeError("SENSITIVE_SENTINEL")
        error.status_code = 429
        backend = Backend([error])
        budget = ExperimentBudget()
        session = AgentSession(self.bank, self.sample, api_budget=budget)

        def configure():
            session.client = backend
            session.model = "mock-model"

        with patch.object(session, "configure_api", side_effect=configure):
            result = session.api()
        self.assertFalse(result["agent_completion"])
        self.assertTrue(budget.unknown_usage)
        with self.assertRaises(ValueError):
            budget.admit([], [], 4096)

    def test_api_placeholder_produces_explicit_failure(self):
        session = AgentSession(self.bank, self.sample)
        with patch.dict(
            "os.environ", {"MODEL_API_KEY": "your-api-key", "MODEL_NAME": "your-tool-calling-model"}
        ):
            result = session.api()
        self.assertFalse(result["agent_completion"])
        self.assertEqual(result["new_api_requests"], 0)


if __name__ == "__main__":
    unittest.main()
