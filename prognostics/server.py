"""Loopback-only research UI; credentials stay on the Python side."""

import io
import json
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .agents import run_prediction
from .cli import load_bank, save_prediction
from .data import PROJECT, online_sample, quality
from .models import POLICIES
from .runtime import experiment_index


def serve(root, model=None, port=8765, open_browser=False):
    if not 1024 <= port <= 65535:
        raise ValueError("端口应在 1024 至 65535 之间。")
    bank, identity = load_bank(model)
    jobs = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200, mime="application/json; charset=utf-8"):
            raw = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, ensure_ascii=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def host_ok(self):
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (
                origin is None or origin in {f"http://{h}" for h in allowed}
            )

        def do_GET(self):
            if not self.host_ok():
                return self.reply({"error": "仅允许本地同源访问。"}, 403)
            parsed = urlparse(self.path)
            q = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    return self.reply(
                        (Path(__file__).parent / "web/index.html").read_bytes(),
                        mime="text/html; charset=utf-8",
                    )
                if parsed.path in {"/app.js", "/style.css"}:
                    return self.reply(
                        (Path(__file__).parent / "web" / parsed.path[1:]).read_bytes(),
                        mime="text/javascript; charset=utf-8"
                        if parsed.path == "/app.js"
                        else "text/css; charset=utf-8",
                    )
                if parsed.path == "/api/overview":
                    result = (
                        (Path(model).parent / "results.json")
                        if model
                        else PROJECT / "artifacts/main-model/results.json"
                    )
                    sweep = PROJECT / "experiments/archive/numerical-pilot/summary.json"
                    validation = PROJECT / "experiments/numerical/tree-groupcv/summary.json"
                    agent_validation = PROJECT / "experiments/agents/paired-api/summary.json"
                    if not agent_validation.exists():
                        agent_validation = PROJECT / "experiments/archive/agent-pilot/summary.json"
                    return self.reply(
                        {
                            "model_manifest": bank.manifest,
                            "experiments": experiment_index(),
                            "research": json.loads(result.read_text(encoding="utf-8")),
                            "sweep": json.loads(sweep.read_text(encoding="utf-8"))
                            if sweep.exists()
                            else None,
                            "validation": json.loads(validation.read_text(encoding="utf-8"))
                            if validation.exists()
                            else None,
                            "agent_validation": json.loads(
                                agent_validation.read_text(encoding="utf-8")
                            )
                            if agent_validation.exists()
                            else None,
                            "roles": [
                                "supervisor",
                                "quality",
                                "numeric",
                                "visual",
                                "text",
                                "maintenance",
                                "retrieval",
                                "uncertainty",
                                "audit",
                            ],
                            "policies": POLICIES,
                            "data": str(root),
                        }
                    )
                if parsed.path == "/api/equipments":
                    return self.reply(
                        [
                            {
                                "equipment": i,
                                "rows": len(np.loadtxt(root / str(i) / "time.txt").reshape(-1)),
                                "split": next(
                                    k for k, v in bank.manifest["devices"].items() if i in v
                                ),
                            }
                            for i in range(1, 51)
                        ]
                    )
                if parsed.path in ["/api/sample", "/api/image"]:
                    e = int(q.get("equipment", ["2"])[0])
                    row = int(q.get("row", ["100"])[0])
                    sample = online_sample(root, e, row)
                    if parsed.path == "/api/sample":
                        return self.reply({"sample": sample, "quality": quality(sample)})
                    with Image.open(root / str(e) / "Image" / f"{row}.tiff") as image:
                        stream = io.BytesIO()
                        image.convert("RGB").save(stream, format="PNG")
                    return self.reply(stream.getvalue(), mime="image/png")
                if parsed.path.startswith("/api/jobs/"):
                    with lock:
                        value = jobs.get(parsed.path.rsplit("/", 1)[1])
                    return self.reply(value or {"error": "任务不存在。"}, 200 if value else 404)
                return self.reply({"error": "未找到。"}, 404)
            except (ValueError, OSError, KeyError) as exc:
                return self.reply({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.host_ok():
                return self.reply({"error": "仅允许本地同源访问。"}, 403)
            if self.path != "/api/predict":
                return self.reply({"error": "未找到。"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("请求大小无效。")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("请求需为 JSON 对象。")
                e = int(body.get("equipment", 2))
                row = int(body.get("row", 100))
                mode = body.get("mode", "local")
                policy = body.get("policy", "learned_router")
                if mode not in ["local", "single", "api"] or policy not in POLICIES:
                    raise ValueError("模式或策略无效。")
                sample = online_sample(root, e, row)
                job = uuid.uuid4().hex
                with lock:
                    if any(j["status"] == "running" for j in jobs.values()):
                        raise ValueError("已有预测任务运行，请等待完成。")
                    jobs[job] = {"status": "running", "mode": mode}

                def worker():
                    try:
                        result = run_prediction(bank, sample, mode, policy, True, identity)
                        path = save_prediction(result)
                        with lock:
                            jobs[job] = {
                                "status": "completed",
                                "result": result,
                                "report_directory": str(path),
                            }
                    except Exception as exc:
                        with lock:
                            jobs[job] = {
                                "status": "failed",
                                "error": str(exc)
                                if isinstance(exc, ValueError)
                                else type(exc).__name__,
                            }

                threading.Thread(target=worker, daemon=True).start()
                return self.reply({"job": job}, 202)
            except (ValueError, OSError, TypeError, KeyError) as exc:
                return self.reply({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"研究实验室：http://127.0.0.1:{port}", flush=True)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
