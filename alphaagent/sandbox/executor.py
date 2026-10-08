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
        start = time.monotonic()
        check = check_alpha_code(code)
        if not check:
            return SandboxResult(False, error="static check failed:\n" + "\n".join(check.errors))

        with tempfile.TemporaryDirectory(prefix="alpha_sbx_") as tmp:
            job = Path(tmp)
            (job / "code.py").write_text(code, encoding="utf-8")
            with open(job / "data.pkl", "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                proc = subprocess.run(
                    [self.python, "-I", str(WORKER), str(job), str(self.memory_mb)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(False, error=f"timeout after {self.timeout:.0f}s", duration=time.monotonic() - start)

            status_file = job / "status.json"
            if not status_file.exists():
                err = (proc.stderr or proc.stdout or "worker crashed")[-2000:]
                return SandboxResult(False, error=f"worker exited with code {proc.returncode}: {err}",
                                     duration=time.monotonic() - start)
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if not status.get("ok"):
                return SandboxResult(False, error=status.get("error", "unknown error"), duration=time.monotonic() - start)
            values = np.load(job / "values.npy", allow_pickle=False)

        series = pd.Series(values, index=data.index, name="alpha")
        return self._validate(series, time.monotonic() - start)

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
