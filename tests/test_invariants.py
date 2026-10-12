"""Tests target causal isolation, group splits, fallback and agent tool boundaries."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from prognostics.agents import AgentSession, Ledger, qualitative_text, run_prediction
from prognostics.data import NUMERIC_NAMES, load_dataset, online_sample
from prognostics.demo import generate_demo
from prognostics.models import ModelBank, group_radius
from prognostics.research import make_split, stress_rows


class CausalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        d = self.root / "1"
        (d / "Numerical").mkdir(parents=True)
        (d / "Image").mkdir()
        np.savetxt(d / "time.txt", [10, 20, 30, 40])
        for name, values in zip(NUMERIC_NAMES, [[5] * 4, [10, 20, 5, 15], [0, 0, 1, 1], [0] * 4]):
            np.savetxt(d / "Numerical" / f"{name}.txt", values)
        (d / "workorder.txt").write_text("stable\nstable\nrepaired\nfuture statement\n")
        for i in range(1, 5):
            im = np.full((20, 20), 255, dtype=np.uint8)
            if i == 2:
                im[5:12, 4:15] = 0
            Image.fromarray(im).save(d / "Image" / f"{i}.tiff")

    def tearDown(self):
        self.tmp.cleanup()

    def test_inference_never_opens_future_label(self):
        original = np.loadtxt

        def guarded(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "nextinslevel.txt")
            return original(path, *args, **kwargs)

        with patch("prognostics.data.np.loadtxt", side_effect=guarded):
            s = online_sample(self.root, 1, 2)
        self.assertNotIn("target", s)

    def test_future_rows_do_not_change_current_features(self):
        before = online_sample(self.root, 1, 2)
        d = self.root / "1"
        (d / "workorder.txt").write_text("stable\nstable\nmalicious future\nchanged future\n")
        np.savetxt(d / "Numerical/nextinstime.txt", [5, 5, 900, 999])
        # Removing unseen images proves that prefix inference does not open them.
        (d / "Image/3.tiff").unlink()
        (d / "Image/4.tiff").unlink()
        self.assertEqual(before, online_sample(self.root, 1, 2))

    def test_repair_invalidates_prior_image_memory(self):
        s = online_sample(self.root, 1, 3)
        self.assertEqual(s["history"][10], 1)
        self.assertEqual(s["history"][11], 1)
        self.assertTrue(any(s["history"][:9]))
        self.assertEqual(s["event_history"][:11], [0] * 11)

    def test_invalid_rows_rejected(self):
        for row in [0, 5]:
            with self.assertRaises(ValueError):
                online_sample(self.root, 1, row)


class ProtocolTests(unittest.TestCase):
    def test_agent_cannot_invent_numeric_explanations(self):
        self.assertEqual(qualitative_text("图像缺失，区间较宽。"), "图像缺失，区间较宽。")
        for text in ["参考区间 88–115 adu", "置信度为９０％", "1. 建议维修"]:
            with self.assertRaises(ValueError):
                qualitative_text(text)

    def test_all_equipment_groups_disjoint(self):
        rows = [{"equipment": i} for i in range(1, 51) for _ in range(3)]
        idx, groups = make_split(rows, 17)
        self.assertEqual([len(groups[k]) for k in groups], [25, 5, 10, 10])
        seen = set()
        for k, ids in groups.items():
            self.assertFalse(seen.intersection(ids))
            seen.update(ids)
            self.assertEqual(set(rows[i]["equipment"] for i in idx[k]), set(ids))
        self.assertEqual(seen, set(range(1, 51)))

    def test_calibration_uses_whole_equipment_maxima(self):
        y = np.arange(1, 21, dtype=float)
        p = np.zeros(20)
        g = np.repeat(np.arange(10), 2)
        radius, maxima = group_radius(y, p, g)
        self.assertEqual(radius, 20)
        self.assertEqual(maxima, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        with self.assertRaises(ValueError):
            group_radius(y[:10], p[:10], g[:10])

    def test_availability_overrides_gate_preference(self):
        row = {
            "numeric": [5, 2, 0, 0],
            "image": [0] * 9,
            "history": [0] * 13,
            "event_history": [0] * 13,
            "text": "stable",
        }
        p = {
            name: np.array([value], dtype=float)
            for name, value in [
                ("numeric_text", 10),
                ("static_fusion", 100),
                ("event_history", 90),
                ("numeric", 5),
                ("numeric_image", 6),
            ]
        }
        gate = SimpleNamespace(predict=lambda x: np.array([[9.0, 0.0, 1.0]]))
        bank = ModelBank()
        bank.routers = {"full": gate}
        self.assertEqual(bank.policy_batch([row], "learned_router", p)[0], 10)
        self.assertEqual(bank.policy_batch([row], "router_unconstrained", p)[0], 100)
        row["text"] = ""
        self.assertEqual(bank.policy_batch([row], "learned_router", p)[0], 5)
        row["image"][0] = 1
        self.assertEqual(bank.policy_batch([row], "learned_router", p)[0], 6)

    def test_stress_test_preserves_original(self):
        row = {
            "image": [1] * 9,
            "history": [1] * 13,
            "event_history": [1] * 13,
            "text": "inspection",
        }
        original = copy.deepcopy(row)
        stressed = stress_rows([row], "image_outage")[0]
        self.assertEqual(row, original)
        self.assertEqual(stressed["event_history"][:11], [0] * 11)


class AgentBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = generate_demo(Path(cls.tmp.name) / "data")
        rows, y = load_dataset(cls.root)
        split, devices = make_split(rows, 17)
        cls.bank = ModelBank(seed=17, trees=8).fit(rows, y, split)
        cls.bank.manifest = {"devices": devices}
        cls.identity = "test-fixture"
        cls.sample = online_sample(cls.root, 2, 20)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.session = AgentSession(self.bank, self.sample)

    def test_arbitrary_tools_and_arguments_rejected(self):
        for name, args in [
            ("read_file", {}),
            ("inspect_quality", {"path": ".env"}),
            ("delegate", []),
            ("delegate", {"role": "shell", "question": "x"}),
        ]:
            with self.assertRaises(ValueError):
                self.session.root_tool(name, args)

    def test_fusion_requires_quality_and_two_experts(self):
        with self.assertRaises(ValueError):
            self.session.root_tool("run_fusion", {})
        self.session.root_tool("inspect_quality", {})
        self.session.completed = {"numeric"}
        with self.assertRaises(ValueError):
            self.session.root_tool("run_fusion", {})
        self.session.completed.add("text")
        self.assertEqual(self.session.root_tool("run_fusion", {}), self.session.forecast)

    def test_audit_requires_fusion(self):
        with self.assertRaises(ValueError):
            self.session.root_tool("verify_forecast", {})
        with self.assertRaises(ValueError):
            self.session.evidence("audit_forecast")

    def test_request_and_token_budget_enforced(self):
        ledger = Ledger(max_requests=2)
        ledger.attempts = 2
        with self.assertRaises(ValueError):
            ledger.limit()
        ledger.attempts = 0
        ledger.calls = [{"charged_completion_tokens": 23500}]
        with self.assertRaises(ValueError):
            ledger.limit()

    def test_failed_unknown_requests_reserve_completion_budget(self):
        ledger = Ledger(completion_budget=4500)
        ledger.attempts = 1
        ledger.failed_completion_reservations = [{"role": "supervisor", "completion_limit": 4096}]
        with self.assertRaises(ValueError):
            ledger.limit()
        self.assertEqual(ledger.summary()["reserved_failed_completion_tokens"], 4096)
        self.assertFalse(ledger.summary()["usage_complete"])

    def test_local_control_has_zero_api_and_exact_model_output(self):
        result = self.session.deterministic()
        self.assertEqual(result["new_api_requests"], 0)
        self.assertEqual(result["mode"], "deterministic_control")
        self.assertEqual(result["forecast"], self.session.forecast)

    def test_retrieval_is_training_only_and_excludes_current_device(self):
        sample = online_sample(self.root, self.bank.manifest["devices"]["train"][0], 20)
        cases = self.bank.retrieve(sample)
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertNotEqual(case["equipment"], sample["equipment"])
            self.assertIn(case["equipment"], self.bank.manifest["devices"]["train"])

    def test_saved_model_roundtrip_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.joblib"
            self.bank.save(path)
            a, b = self.bank.forecast(self.sample), ModelBank.load(path).forecast(self.sample)
            self.assertEqual(a["selected_expert"], b["selected_expert"])
            self.assertAlmostEqual(a["prediction_adu"], b["prediction_adu"], places=10)
            np.testing.assert_allclose(a["interval_adu"], b["interval_adu"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                list(a["expert_predictions"].values()),
                list(b["expert_predictions"].values()),
                rtol=0,
                atol=1e-10,
            )

    def test_successful_api_cache_never_reissues_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("prognostics.agents.PROJECT", Path(tmp)),
                patch.object(
                    AgentSession, "api", return_value=self.session.result("api", "mock", None)
                ) as api,
            ):
                first = run_prediction(self.bank, self.sample, "api", model_identity="test")
                second = run_prediction(self.bank, self.sample, "api", model_identity="test")
        self.assertEqual(api.call_count, 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(second["new_api_requests"], 0)


if __name__ == "__main__":
    unittest.main()
