import http.client
import json
import socket
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prognostics.server import serve


class ServerTests(unittest.TestCase):
    def test_http_prediction_modes_and_invalid_mode(self):
        ready = threading.Event()
        servers = []
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        def create_server(address, handler):
            server = ThreadingHTTPServer(address, handler)
            servers.append(server)
            ready.set()
            return server

        def predict(bank, sample, mode, *args):
            return {"mode": mode, "agent_completion": True}

        def request(method, path, body=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request(
                    method,
                    path,
                    body=None if body is None else json.dumps(body),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        with (
            patch("prognostics.server.ThreadingHTTPServer", side_effect=create_server),
            patch("prognostics.server.load_bank", return_value=(SimpleNamespace(), "test")),
            patch("prognostics.server.online_sample", return_value={"equipment": 2, "row": 8}),
            patch("prognostics.server.run_prediction", side_effect=predict) as run,
            patch("prognostics.server.save_prediction", return_value=Path("runs/test-http")),
        ):
            thread = threading.Thread(target=serve, args=(Path("data/demo"), None, port))
            thread.start()
            try:
                self.assertTrue(ready.wait(5), "Server did not start")
                for mode in ("local", "single", "api"):
                    status, accepted = request(
                        "POST", "/api/predict", {"equipment": 2, "row": 8, "mode": mode}
                    )
                    self.assertEqual(status, 202)
                    deadline = time.monotonic() + 5
                    while True:
                        status, job = request("GET", "/api/jobs/" + accepted["job"])
                        self.assertEqual(status, 200)
                        if job["status"] != "running":
                            break
                        self.assertLess(time.monotonic(), deadline, "Job did not complete")
                        time.sleep(0.01)
                    self.assertEqual(job["status"], "completed")
                    self.assertEqual(job["result"]["mode"], mode)
                status, _ = request("POST", "/api/predict", {"mode": "unsupported"})
                self.assertEqual(status, 400)
                self.assertEqual(run.call_count, 3)
            finally:
                if servers:
                    servers[0].shutdown()
                thread.join(5)
                self.assertFalse(thread.is_alive())
