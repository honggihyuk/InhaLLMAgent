"""AST policy for LLM-generated alpha code: no imports, no I/O, no introspection, one entry point."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
    "getattr", "setattr", "delattr", "help", "exit", "quit", "memoryview", "os", "sys", "subprocess",
    "shutil", "socket", "pickle", "importlib", "builtins", "ctypes",
}
# pandas/numpy entry points that read/write files, run code, or reach the network
FORBIDDEN_ATTRS = {
    "read_csv", "read_parquet", "read_pickle", "read_json", "read_sql", "read_html", "read_excel", "read_table",
    "read_feather", "read_hdf", "to_csv", "to_parquet", "to_pickle", "to_json", "to_sql", "to_excel", "to_hdf",
    "to_feather", "load", "save", "savez", "savetxt", "loadtxt", "fromfile", "tofile", "memmap", "system",
    "popen", "eval", "query", "ctypeslib",
}


@dataclass
class StaticCheckResult:
    ok: bool
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def check_alpha_code(code: str, entry_point: str = "compute_alpha", max_chars: int = 20_000) -> StaticCheckResult:
    errors: List[str] = []
    if len(code) > max_chars:
        return StaticCheckResult(False, [f"code too long ({len(code)} chars)"])
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return StaticCheckResult(False, [f"SyntaxError: {e.msg} (line {e.lineno})"])

    defined = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(f"line {node.lineno}: imports are not allowed (pd and np are provided)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                errors.append(f"line {node.lineno}: dunder attribute '{node.attr}' is not allowed")
            elif node.attr in FORBIDDEN_ATTRS:
                errors.append(f"line {node.lineno}: '.{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append(f"line {node.lineno}: global/nonlocal are not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            errors.append(f"line {node.lineno}: async/generators are not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == entry_point:
            defined = True
            if len(node.args.args) != 1:
                errors.append(f"{entry_point} must take exactly one argument (df)")

    # top level may only hold function definitions, constant assignments, and docstrings
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        errors.append(f"line {node.lineno}: only function definitions are allowed at module level")
    if not defined:
        errors.append(f"no function named {entry_point} was defined")
    return StaticCheckResult(not errors, errors)
