"""Builds (once, cached under build/mcts_cpp) and returns the C++ search forest of forest.cpp."""
import os
import sys
from pathlib import Path

_MODULE = None


def forest_module():
    global _MODULE
    if _MODULE is None:
        from torch.utils import cpp_extension

        # ninja lives next to the interpreter (venv/bin), which is not on PATH when venv/bin/python is run directly
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        here = Path(__file__).resolve().parent
        build = here.parents[1] / "build" / "mcts_cpp"
        build.mkdir(parents=True, exist_ok=True)
        try:
            _MODULE = cpp_extension.load(
                name="treerm_mcts_forest", sources=[str(here / "forest.cpp")], extra_cflags=["-O3"],
                build_directory=str(build), verbose=bool(os.environ.get("MCTS_CPP_VERBOSE")),
            )
        except Exception as exc:  # no compiler / headers: say so instead of a wall of ninja output
            raise ImportError(f"mcts_impl=cpp needs a C++ toolchain to build {here / 'forest.cpp'} (g++, ninja): {exc}") from exc
    return _MODULE
