"""Optional portable PyTorch regressors for the existing causal modality feature banks."""

import numpy as np


def torch_runtime():
    try:
        import torch
    except ImportError:
        raise ValueError("神经网络后端需要 PyTorch，请安装项目的 neural 可选依赖。") from None
    return torch


def resolve_device(requested="auto"):
    torch = torch_runtime()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("设备需为 auto、cpu、cuda 或 mps。")
    cuda = torch.cuda.is_available()
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise ValueError(
            "CUDA 不可用，请检查 NVIDIA 驱动及 PyTorch CUDA 构建，或使用 --device cpu。"
        )
    if requested == "mps" and not mps:
        raise ValueError("MPS 不可用，请使用支持 MPS 的 Mac 与 PyTorch，或使用 --device cpu。")
    return requested


class NeuralRegressor:
    """Train-only normalization, fixed-epoch MLP, and NumPy weight serialization."""

    def __init__(self, seed=17, epochs=80, batch_size=512, device="auto"):
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs 与 batch_size 必须为正整数。")
        self.seed, self.epochs, self.batch_size = seed, epochs, batch_size
        self.device = device
        self._network = None

    def network(self):
        torch = torch_runtime()
        return torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 1),
        )

    def fit(self, x, y):
        torch = torch_runtime()
        self.training_device = resolve_device(self.device)
        if self.training_device == "cpu":
            torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("神经网络训练输入包含非有限值。")
        self.input_size = x.shape[1]
        self.x_mean, self.x_scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
        self.y_mean, self.y_scale = float(y.mean()), max(float(y.std()), 1e-6)
        tx = torch.as_tensor((x - self.x_mean) / self.x_scale, device=self.training_device)
        ty = torch.as_tensor((y - self.y_mean) / self.y_scale, device=self.training_device)
        model = self.network().to(self.training_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        rng = np.random.default_rng(self.seed)
        self.loss_curve = []
        model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(x))
            total = 0.0
            for start in range(0, len(x), self.batch_size):
                idx = torch.as_tensor(
                    order[start : start + self.batch_size], device=self.training_device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(tx[idx]).squeeze(-1), ty[idx])
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(idx)
            self.loss_curve.append(total / len(x))
        self.weights = {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in model.state_dict().items()
        }
        self._network = model.eval()
        self.inference_device = self.training_device
        return self

    def predict(self, x):
        torch = torch_runtime()
        if self._network is None:
            # Checkpoints carry CPU NumPy arrays and can move between CUDA, MPS and CPU hosts.
            self.inference_device = resolve_device("auto")
            self._network = self.network()
            self._network.load_state_dict(
                {n: torch.from_numpy(w.copy()) for n, w in self.weights.items()}
            )
            self._network = self._network.to(self.inference_device).eval()
        x = np.asarray(x, dtype=np.float32)
        scaled = (x - self.x_mean) / self.x_scale
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(x), 2048):
                tx = torch.as_tensor(scaled[start : start + 2048], device=self.inference_device)
                predictions.append(self._network(tx).squeeze(-1).cpu().numpy())
        return np.concatenate(predictions).astype(np.float64) * self.y_scale + self.y_mean

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_network"] = None
        return state
