from __future__ import annotations

import os
from pathlib import Path

from dasbench.utils import timestamp_token

DEFAULT_DATASETS_ROOT = Path("artifacts/datasets")
DEFAULT_AGENT_RUNS_ROOT = Path("artifacts/agent_runs")
DEFAULT_REPORTS_ROOT = Path("artifacts/reports")

MAIN_SWEEP_ROOT_ENV_VAR = "MAIN_SWEEP_ROOT"
# The sweep the paper reports. Analyses default to it so published numbers
# reproduce byte for byte; point MAIN_SWEEP_ROOT at your own main-benchmark
# sweep to run the same analyses over a fresh run.
PAPER_MAIN_SWEEP_ROOT = Path("artifacts/second_scale_benchmark_v2/20260427_230552")
MAIN_SWEEP_CONDITION_ID = "seconds_scale_v2"


def main_sweep_root() -> Path:
    """Root of the main benchmark sweep that the analysis scripts read."""
    configured = os.environ.get(MAIN_SWEEP_ROOT_ENV_VAR, "").strip()
    return Path(configured) if configured else PAPER_MAIN_SWEEP_ROOT


def default_dataset_dir(problem: str, family: str, dataset_id: str | None = None) -> Path:
    resolved_id = dataset_id or timestamp_token()
    return DEFAULT_DATASETS_ROOT / problem / family / resolved_id


def default_agent_run_dir(problem: str, family: str, run_id: str | None = None) -> Path:
    resolved_id = run_id or timestamp_token()
    return DEFAULT_AGENT_RUNS_ROOT / problem / family / resolved_id


def default_report_dir(problem: str, family: str, run_id: str) -> Path:
    return DEFAULT_REPORTS_ROOT / problem / family / run_id
