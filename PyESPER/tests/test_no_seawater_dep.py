"""Guard against the deprecated CSIRO ``seawater`` package creeping back in.

``24c63cd`` replaced ``seawater`` with the in-repo njit'd EOS-80 in
:mod:`PyESPER.eos80_jit`, but converted only ``iterations.py`` and
``concurrency.py``; ``pH_adjustment.py`` and ``pH_DIC_nn_adjustment.py`` were
missed and kept importing the real package for another five months. Because
those imports are *function-local*, importing the package does not trip them --
they only fire once a caller asks for pH or DIC with ``EstDates``, which is why
this went unnoticed until it raised ``ModuleNotFoundError`` on a machine that
happened not to have ``seawater`` installed.

So the primary check here is a source-level scan, not an import: it catches a
dormant import in a branch no test exercises, and it works whether or not
``seawater`` is installed in the running environment.
"""

import ast
import pathlib

import numpy as np
import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def _imported_names(path):
    """Top-level module names imported by ``path``, at any nesting depth.

    Parsed rather than grepped so that the many prose mentions of "seawater
    properties" in the docstrings (lir.py, mixed.py, nn.py, errors.py ...) and
    the ``# import seawater as sw`` comments left as migration breadcrumbs do
    not register as usage.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def test_no_module_imports_seawater():
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in sorted(PACKAGE_ROOT.rglob("*.py"))
        if "seawater" in set(_imported_names(p))
    ]
    assert offenders == [], (
        "the deprecated `seawater` package is imported by: "
        + ", ".join(offenders)
        + " -- use PyESPER.eos80_jit instead (same EOS-80 formulation, njit'd)"
    )


@pytest.mark.parametrize("name", ["requirements.txt", "setup.py", "environment.yml"])
def test_seawater_is_not_a_declared_dependency(name):
    """Nothing should pull the package back into an environment.

    Commented-out mentions are fine and are deliberately kept as a record of
    what was dropped, so only uncommented lines are considered.
    """
    path = REPO_ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    live = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "seawater" in line and not line.strip().lstrip("-").strip().startswith("#")
    ]
    assert live == [], f"{name} still declares seawater: {live}"


def test_eos80_jit_pres_matches_the_package_it_replaced():
    """The replacement must be bit-identical, not merely close.

    Skipped where ``seawater`` is absent -- which is the normal, desired state.
    It is only installed here as the oracle for this comparison.
    """
    sw = pytest.importorskip(
        "seawater", reason="seawater not installed (expected); nothing to compare against"
    )
    from PyESPER import eos80_jit

    rng = np.random.default_rng(0)
    depth = rng.uniform(0.0, 6000.0, 2000)
    lat = rng.uniform(-80.0, 80.0, 2000)

    np.testing.assert_array_equal(eos80_jit.pres(depth, lat), sw.pres(depth, lat))
