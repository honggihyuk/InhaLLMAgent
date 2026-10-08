"""Sandbox child process, run as ``python -I <this file> <job_dir> <mem_mb> <n_datasets>``.

Reads ``code.py`` and ``data_<i>.pkl`` (written by the trusted parent) from the job directory, runs
``compute_alpha`` on each dataset with a reduced builtins table, and reports back through files the parent
can read without unpickling anything the untrusted code could have produced:

* ``values_<i>.npy`` - float64 array aligned to dataset i's index (loaded with ``allow_pickle=False``)
* ``status.json``    - {"ok": bool, "error": str}
"""

import json
import pickle
import sys
import traceback
from pathlib import Path

import builtins as _b

_SAFE = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len",
    "list", "map", "max", "min", "pow", "range", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError",
    "Exception", "__build_class__", "object", "property", "staticmethod", "classmethod", "super", "print",
)
SAFE_BUILTINS = {name: getattr(_b, name) for name in _SAFE}


def _limit_resources(mem_mb: int) -> None:
    try:
        import resource  # POSIX only; on Windows the parent's timeout is the guard

        limit = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def main(job_dir: str, mem_mb: int, n: int) -> None:
    job = Path(job_dir)
    code = (job / "code.py").read_text(encoding="utf-8")
    datasets = []
    for i in range(n):
        with open(job / f"data_{i}.pkl", "rb") as fh:
            datasets.append(pickle.load(fh))
    import numpy as np
    import pandas as pd

    status = {"ok": False, "error": ""}
    _limit_resources(mem_mb)
    try:
        compiled = compile(code, "<alpha>", "exec")
        for i, df in enumerate(datasets):
            # fresh namespace per dataset: no state can carry over between runs
            namespace = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "__name__": "alpha"}
            exec(compiled, namespace)
            out = namespace["compute_alpha"](df.copy())
            if isinstance(out, pd.DataFrame):
                if out.shape[1] != 1:
                    raise ValueError(f"compute_alpha returned a DataFrame with {out.shape[1]} columns")
                out = out.iloc[:, 0]
            if not isinstance(out, pd.Series):
                raise TypeError(f"compute_alpha must return a pd.Series, got {type(out).__name__}")
            if not out.index.equals(df.index):
                if out.index.nlevels != df.index.nlevels or out.index.has_duplicates:
                    raise ValueError("compute_alpha output index does not match the input (date, ticker) index")
                out = out.reindex(df.index)
            values = pd.to_numeric(out, errors="coerce").to_numpy(dtype="float64")
            np.save(job / f"values_{i}.npy", values, allow_pickle=False)
        status["ok"] = True
    except Exception:
        status["error"] = traceback.format_exc(limit=4)[-2000:]
    (job / "status.json").write_text(json.dumps(status), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2048, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
