import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prognostics.cli import stamp
from prognostics.runtime import doctor_report

LAUNCH = Path(__file__).resolve().parent.parent / "scripts/launch.py"
spec = importlib.util.spec_from_file_location("launcher", LAUNCH)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PortabilityTests(unittest.TestCase):
    def test_windows_and_macos_venv_paths(self):
        self.assertEqual(
            launcher.interpreter(Path("project"), True).as_posix(),
            "project/.venv/Scripts/python.exe",
        )
        self.assertEqual(
            launcher.interpreter(Path("project"), False).as_posix(), "project/.venv/bin/python"
        )

    def test_timestamp_has_no_external_timezone_database_dependency(self):
        self.assertRegex(stamp(), r"^\d{8}-\d{6}-\d{6}$")

    def test_doctor_masks_api_key_and_url_credentials(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("prognostics.runtime.PROJECT", Path(directory)),
            patch.dict(
                "os.environ",
                {
                    "MODEL_API_KEY": "PRIVATE_SENTINEL",
                    "MODEL_NAME": "example",
                    "MODEL_BASE_URL": "https://user:PRIVATE_SENTINEL@example.test/v1",
                },
            ),
        ):
            report = doctor_report("cpu")
        self.assertTrue(report["api_configured"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(report))
        self.assertEqual(report["api_host"], "example.test")
