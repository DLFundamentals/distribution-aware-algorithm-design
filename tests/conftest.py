"""Test-suite guards.

The suite is meant to run offline. Gurobi is the one exception: a WLS license
checks out over the network on every environment creation, so the tests that
build a real Gurobi solver carry the `gurobi` marker and are deselected by
default. Nothing else in the suite contacts an external service -- the LLM
tests use fake base URLs and mocked clients, and the PACE tests read instances
from a local directory rather than downloading them.
"""

from __future__ import annotations

import functools

import pytest


@functools.lru_cache(maxsize=1)
def _gurobi_license() -> tuple[bool, str]:
    """Probe once per session, and only when a gurobi-marked test is selected."""
    try:
        import gurobipy as gp
    except ImportError:
        return False, "gurobipy is not installed"
    try:
        env = gp.Env(params={"OutputFlag": 0})
    except Exception as exc:  # noqa: BLE001 - any license failure means skip, not fail
        return False, f"no usable Gurobi license: {type(exc).__name__}: {exc}"
    env.dispose()
    return True, ""


@pytest.fixture(autouse=True)
def _require_gurobi_license(request: pytest.FixtureRequest) -> None:
    """Skip, rather than fail, when a gurobi-marked test has no license.

    This runs at setup, after -m deselection, so a default offline run never
    reaches the probe and never touches the license service.
    """
    if request.node.get_closest_marker("gurobi") is None:
        return
    available, reason = _gurobi_license()
    if not available:
        pytest.skip(reason)
