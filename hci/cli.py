r"""Console entry points — `uv run train`, `uv run test`, `uv run infer`"""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _run(script: str) -> None:
    # Run the file by path instead of importing it by name: a module called `test` would resolve
    # to the standard library's test package on Pythons that ship it.
    runpy.run_path(str(_ROOT / script), run_name="__main__")


def train() -> None:
    _run("train.py")


def test() -> None:
    _run("test.py")


def infer() -> None:
    _run("infer.py")
