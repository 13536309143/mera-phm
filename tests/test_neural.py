import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np

from prognostics.models import ModelBank
from prognostics.neural import NeuralRegressor, resolve_device


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        cls.x = rng.normal(size=(160, 4)).astype(np.float32)
        cls.y = 12 + cls.x[:, 0] * 3 - cls.x[:, 1] * 2
        cls.model = NeuralRegressor(epochs=45, batch_size=80, device="cpu").fit(cls.x, cls.y)

    def test_cpu_learning_and_finite_predictions(self):
        self.assertLess(self.model.loss_curve[-1], self.model.loss_curve[0] * 0.3)
        self.assertLess(np.mean(np.abs(self.model.predict(self.x) - self.y)), 1.0)

    def test_normalization_never_refits_on_prediction_inputs(self):
        mean = self.model.x_mean.copy()
        self.model.predict(self.x + 100)
        np.testing.assert_array_equal(self.model.x_mean, mean)
        np.testing.assert_allclose(mean, self.x.mean(axis=0))

    def test_saved_weights_are_numpy_and_checkpoint_is_portable(self):
        self.assertIsNone(self.model.__getstate__()["_network"])
        self.assertTrue(all(isinstance(w, np.ndarray) for w in self.model.weights.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(self.model, path)
            restored = joblib.load(path)
            self.assertIsNone(restored._network)
            original, replay = self.model.predict(self.x), restored.predict(self.x)
            if not np.allclose(original, replay, atol=1e-4):
                import torch

                left = torch.from_numpy((self.x - self.model.x_mean) / self.model.x_scale)
                right = left.to(restored.inference_device)
                with torch.inference_mode():
                    for i, (cpu_layer, restored_layer) in enumerate(
                        zip(self.model._network, restored._network)
                    ):
                        left, right = cpu_layer(left), restored_layer(right)
                        print(
                            "portable layer",
                            i,
                            type(cpu_layer).__name__,
                            "max difference",
                            float((left - right.cpu()).abs().max()),
                        )
                        if hasattr(cpu_layer, "weight"):
                            print(
                                "weight difference",
                                float((cpu_layer.weight - restored_layer.weight.cpu()).abs().max()),
                            )
                            print(
                                "bias difference",
                                float((cpu_layer.bias - restored_layer.bias.cpu()).abs().max()),
                            )
            np.testing.assert_allclose(original, replay, atol=1e-4)

    def test_unavailable_gpu_rejected_and_auto_falls_back(self):
        with patch(
            "prognostics.neural.torch_runtime",
            return_value=SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            ),
        ):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaises(ValueError):
                resolve_device("cuda")
            with self.assertRaises(ValueError):
                resolve_device("mps")

    def test_available_accelerator_executes_training(self):
        device = resolve_device("auto")
        if device == "cpu":
            self.skipTest("No accelerator on this host")
        model = NeuralRegressor(epochs=3, device=device).fit(self.x, self.y)
        self.assertEqual(model.training_device, device)
        self.assertTrue(np.isfinite(model.predict(self.x)).all())


class BackendContractTests(unittest.TestCase):
    def test_tree_backend_rejects_gpu_flag(self):
        with self.assertRaises(ValueError):
            ModelBank(backend="trees", device="cuda")


if __name__ == "__main__":
    unittest.main()
