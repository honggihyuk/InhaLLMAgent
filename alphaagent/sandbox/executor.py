"""Safe execution of LLM-generated alpha code.

Layers: (1) AST policy check, (2) separate isolated interpreter process (``python -I``) with a reduced
builtins table, (3) wall-clock timeout and, on POSIX, an address-space limit, (4) results returned as a
plain float array (never unpickled from the child), (5) output validation (shape, finiteness, coverage).
For production, run the worker inside a locked-down container (no network, read-only FS); see deploy/.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from alphaagent.sandbox.static_check import check_alpha_code

WORKER = Path(__file__).with_name("_worker.py")


@dataclass
class SandboxResult:
    ok: bool
    values: Optional[pd.Series] = None
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0


class AlphaSandbox:
    def __init__(
        self,
        timeout: float = 60.0,
        memory_mb: int = 2048,
        min_coverage: float = 0.3,
        python: Optional[str] = None,
    ) -> None:
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.min_coverage = min_coverage
        self.python = python or sys.executable

    def run(self, code: str, data: pd.DataFrame) -> SandboxResult:
        return self.run_many(code, [data])[0]

    def run_many(self, code: str, datasets: List[pd.DataFrame]) -> List[SandboxResult]:
        """Evaluate the same code on several datasets in ONE isolated process (each with a fresh namespace).
        Saves interpreter start-up for look-ahead truncation tests. Only the first result is output-validated."""
        start = time.monotonic()

        def fail(msg: str) -> List[SandboxResult]:
            return [SandboxResult(False, error=msg, duration=time.monotonic() - start) for _ in datasets]

        check = check_alpha_code(code)
        if not check:
            return fail("static check failed:\n" + "\n".join(check.errors))

        with tempfile.TemporaryDirectory(prefix="alpha_sbx_") as tmp:
            job = Path(tmp)
            (job / "code.py").write_text(code, encoding="utf-8")
            for i, data in enumerate(datasets):
                with open(job / f"data_{i}.pkl", "wb") as fh:
                    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                proc = subprocess.run(
                    [self.python, "-I", str(WORKER), str(job), str(self.memory_mb), str(len(datasets))],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * len(datasets),
                )
            except subprocess.TimeoutExpired:
                return fail(f"timeout after {self.timeout * len(datasets):.0f}s")

            status_file = job / "status.json"
            if not status_file.exists():
                err = (proc.stderr or proc.stdout or "worker crashed")[-2000:]
                return fail(f"worker exited with code {proc.returncode}: {err}")
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if not status.get("ok"):
                return fail(status.get("error", "unknown error"))
            values = [np.load(job / f"values_{i}.npy", allow_pickle=False) for i in range(len(datasets))]

        duration = time.monotonic() - start
        results = [self._validate(pd.Series(values[0], index=datasets[0].index, name="alpha"), duration)]
        for v, data in zip(values[1:], datasets[1:]):
            s = pd.Series(v, index=data.index, name="alpha").replace([np.inf, -np.inf], np.nan)
            results.append(SandboxResult(True, s, duration=duration))
        return results

    def _validate(self, series: pd.Series, duration: float) -> SandboxResult:
        warnings: List[str] = []
        n_inf = int(np.isinf(series).sum())
        if n_inf:
            warnings.append(f"{n_inf} infinite values replaced with NaN")
            series = series.replace([np.inf, -np.inf], np.nan)
        coverage = float(series.notna().mean()) if len(series) else 0.0
        if coverage < self.min_coverage:
            return SandboxResult(False, series, f"alpha covers only {coverage:.0%} of rows (min {self.min_coverage:.0%})",
                                 warnings, duration)
        per_date_unique = series.groupby(level="date").nunique()
        if (per_date_unique <= 1).mean() > 0.9:
            return SandboxResult(False, series, "alpha is constant across tickers on most dates (no cross-sectional signal)",
                                 warnings, duration)
        return SandboxResult(True, series, "", warnings, duration)
