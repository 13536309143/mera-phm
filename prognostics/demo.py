"""Original toy data for installation tests; not the paper's simulator or benchmark."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import NUMERIC_NAMES


def generate_demo(root, seed=17, devices=50, rows=24):
    root = Path(root)
    if root.exists():
        raise ValueError("Demo output already exists; choose a new directory.")
    if devices != 50 or rows < 20:
        raise ValueError("The demo protocol requires 50 devices and at least 20 rows.")
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for equipment in range(1, devices + 1):
        directory = root / str(equipment)
        (directory / "Numerical").mkdir(parents=True)
        (directory / "Image").mkdir()
        times = np.arange(1, rows + 1) * 5.0
        level = float(rng.uniform(8, 30))
        rate = float(rng.uniform(0.15, 0.55))
        last_repair = 0.0
        chemical = 0
        mechanical = 0
        numeric = []
        texts = []
        targets = []
        for j, t in enumerate(times):
            if j in [9, 18]:
                level *= 0.45
                last_repair = t
                chemical += 1
            level += rate * 5
            horizon = int(rng.integers(1, 51))
            numeric.append([horizon, t - last_repair, chemical, mechanical])
            description = "equipment stable" if level < 22 else "moderate degradation observed"
            if j in [9, 18]:
                description = "chemical cleaning performed; equipment restored"
            texts.append(description)
            targets.append(level + rate * horizon + rng.normal(0, 0.8))
            image = Image.new("L", (96, 64), 255)
            draw = ImageDraw.Draw(image)
            draw.line([(4, 2), (4, 58), (92, 58)], fill=0, width=1)
            if j > 0 and rng.random() > 0.25:
                points = [
                    (x, int(np.clip(57 - level * 0.7 + 2 * np.sin(x / 7), 3, 56)))
                    for x in range(7, 90)
                ]
                draw.line(points, fill=0, width=1)
            image.save(directory / "Image" / f"{j + 1}.tiff")
        np.savetxt(directory / "time.txt", times)
        np.savetxt(directory / "nextinslevel.txt", targets)
        for i, name in enumerate(NUMERIC_NAMES):
            np.savetxt(directory / "Numerical" / f"{name}.txt", np.array(numeric)[:, i])
        (directory / "workorder.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    (root / "README.txt").write_text(
        "Original MERA-PHM software-test fixture. This is not the paper dataset or its simulator. Do not use toy results as evidence of industrial predictive accuracy.\n"
    )
    return root
