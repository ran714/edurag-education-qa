"""Run offline regressions and save the actual test count and failures."""

import json
import os
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["EDURAG_MODE"] = "demo"
os.environ["EDURAG_SESSION_DB"] = ":memory:"

if __name__ == "__main__":
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "seconds": time.perf_counter() - started,
        "successful": result.wasSuccessful(),
        "scope": "offline_regression_with_test_doubles_not_real_provider",
    }
    path = ROOT / "reports/regression.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
