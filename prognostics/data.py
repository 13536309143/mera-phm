"""Separate causal inference observations from offline future labels."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(os.getenv("MERA_WORKSPACE", Path.cwd())).resolve()
NUMERIC_NAMES = [
    "nextinstime",
    "timefromlastrepair",
    "Numberofchemicalcleaning",
    "Numberofmechanicalcleaning",
]


def discover_data(explicit=None):
    if explicit is None:
        from dotenv import dotenv_values

        explicit = os.getenv("DATASET_PATH") or dotenv_values(PROJECT / ".env").get("DATASET_PATH")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "1/time.txt").is_file():
            raise ValueError("数据目录中没有 1/time.txt。")
        return root
    choices = [
        PROJECT / "data/raw/Multimodal data for prognostics",
        PROJECT / "Multimodal data for prognostics",
        PROJECT / "data/demo",
    ]
    for root in choices:
        if (root / "1/time.txt").is_file():
            return root
    raise ValueError("找不到数据集，请通过 --data 指定包含 1 至 50 目录的数据根目录。")


def image_descriptor(array, template):
    marks = (array < 128) & (template >= 128)
    ys, xs = np.where(marks)
    if not len(xs):
        return np.zeros(9, dtype=np.float32)
    yn, xn = ys / (array.shape[0] - 1), xs / (array.shape[1] - 1)
    return np.array(
        [
            1,
            len(xs) / array.size,
            yn.mean(),
            yn.std(),
            yn.min(),
            yn.max(),
            xn.max() - xn.min(),
            xn.mean(),
            xn.std(),
        ],
        dtype=np.float32,
    )


def clean_text(text):
    return " ".join(text.strip().strip('"').lower().split())


def observations(root, equipment, upto=None):
    """This function never opens nextinslevel.txt. All histories stop at the requested row."""
    if not 1 <= int(equipment) <= 50:
        raise ValueError("设备编号应在 1 至 50 之间。")
    directory = Path(root) / str(int(equipment))
    times = np.loadtxt(directory / "time.txt").reshape(-1)
    count = len(times) if upto is None else int(upto)
    if not 1 <= count <= len(times):
        raise ValueError(f"采样行应在 1 至 {len(times)} 之间。")
    numeric = np.column_stack(
        [np.loadtxt(directory / "Numerical" / f"{name}.txt").reshape(-1) for name in NUMERIC_NAMES]
    )[:count]
    texts = (directory / "workorder.txt").read_text().splitlines()[:count]
    times = times[:count]
    if len(numeric) != count or len(texts) != count or not np.all(np.diff(times) > 0):
        raise ValueError("数值、文本与时间未对齐或时间不递增。")
    if not np.isfinite(numeric).all() or not np.isfinite(times).all() or np.any(numeric[:, 0] <= 0):
        raise ValueError("存在无效数值或非正预测跨度。")
    with Image.open(Path(root) / "1/Image/1.tiff") as image:
        template = np.asarray(image.convert("L"))
    last_image = np.zeros(9, dtype=np.float32)
    last_image_time = None
    last_text_time = float(times[0])
    last_text = None
    result = []
    for i, t in enumerate(times):
        with Image.open(directory / "Image" / f"{i + 1}.tiff") as image:
            array = np.asarray(image.convert("L"))
        if array.shape != template.shape:
            raise ValueError("图片尺寸与固定坐标轴模板不一致。")
        im = image_descriptor(array, template)
        if im[0]:
            last_image, last_image_time = im.copy(), float(t)
        text = clean_text(texts[i])
        if text != last_text:
            last_text_time = float(t)
        last_text = text
        age = 0 if last_image_time is None else float(t) - last_image_time
        predates = float(last_image_time is not None and last_image_time < float(t - numeric[i, 1]))
        history = np.concatenate(
            [
                last_image,
                [age, float(last_image_time is not None), predates, float(t) - last_text_time],
            ]
        )
        event_history = history.copy()
        if predates:
            # A pre-maintenance image is not carried over as current post-maintenance evidence.
            event_history[:11] = 0
        result.append(
            {
                "equipment": int(equipment),
                "row": i + 1,
                "time": float(t),
                "numeric": numeric[i].tolist(),
                "image": im.tolist(),
                "history": history.tolist(),
                "event_history": event_history.tolist(),
                "text": text,
            }
        )
    return result


def online_sample(root, equipment, row):
    return observations(root, equipment, row)[-1]


def load_dataset(root):
    """Offline training/evaluation loader; labels remain outside observation dictionaries."""
    rows, labels = [], []
    for equipment in range(1, 51):
        part = observations(root, equipment)
        y = np.loadtxt(Path(root) / str(equipment) / "nextinslevel.txt").reshape(-1)
        if len(y) != len(part) or not np.isfinite(y).all():
            raise ValueError("未来标签与观察记录不一致。")
        rows.extend(part)
        labels.extend(y)
    return rows, np.asarray(labels, dtype=np.float64)


def quality(sample):
    n, im, h = sample["numeric"], sample["image"], sample["history"]
    flags = []
    if not im[0]:
        flags.append("当前没有可见图像曲线")
    if h[11]:
        flags.append("最近图像早于最近维护，事件记忆已屏蔽该图像")
    if h[12] > 25:
        flags.append("巡检文本长期未变化；年龄是代理量")
    if n[0] > 25:
        flags.append("预测跨度较长")
    return {
        "image_available": bool(im[0]),
        "image_age_atu": h[9],
        "image_predates_repair": bool(h[11]),
        "text_change_age_proxy_atu": h[12],
        "horizon_atu": n[0],
        "flags": flags,
        "label_access": False,
    }


def sample_hash(sample):
    return hashlib.sha256(
        json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def dataset_fingerprint(root):
    """Offline provenance only; never invoked from the prediction endpoint."""
    digest = hashlib.sha256()
    count = 0
    for equipment in range(1, 51):
        for path in sorted((Path(root) / str(equipment)).rglob("*")):
            if (
                not path.is_file()
                or path.name.startswith("._")
                or path.suffix.lower() not in [".txt", ".tiff"]
            ):
                continue
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            count += 1
    return {
        "sha256": digest.hexdigest(),
        "files": count,
        "scope": "all device .txt and .tiff files, including offline labels",
    }
