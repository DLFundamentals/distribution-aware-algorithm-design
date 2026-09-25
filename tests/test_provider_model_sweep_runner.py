from __future__ import annotations

import json
from pathlib import Path

from scripts.run_provider_model_sweep import (
    Provider,
    build_parser,
    build_stages,
    run,
)


def test_main_llm_pv_stage_runs_all_families(tmp_path: Path) -> None:
    provider = Provider("kimi-k26", Path(".env.kimi-k26"))
    stages = build_stages(
        provider,
        tmp_path,
        main_max_workers=1,
        llm_pv_max_workers=1,
        llm_pv_attempts=5,
        force=False,
    )
    stage = next(item for item in stages if item.name == "main21_llm_pv")

    assert "--all-families" in stage.command
    assert "--source-condition-id" in stage.command
    assert "seconds_scale_v2" in stage.command
    assert stage.completion_path == tmp_path / "kimi-k26" / "llm_pv_benchmark" / "kimi-k26_main21_llm_pv" / "benchmark_sweep_summary.json"


def test_pace_imports_llm_pv_uses_provider_datasets(tmp_path: Path) -> None:
    provider = Provider("deepseek-v4-pro", Path(".env.deepseek-v4-pro"))
    stages = build_stages(
        provider,
        tmp_path,
        main_max_workers=1,
        llm_pv_max_workers=1,
        llm_pv_attempts=5,
        force=False,
    )
    stage = next(item for item in stages if item.name == "pace_imports_llm_pv")
    command_text = " ".join(str(part) for part in stage.command)

    assert "benchmarks.pace_llm_pv_baselines" in command_text
    assert "pace2025_hs=" in command_text
    assert "pace2024_ocm_exact=" in command_text
    assert "pace2024_ocm_cutwidth=" in command_text
    assert "pace2022_dfvs_heuristic=" in command_text
    assert str(tmp_path / "deepseek-v4-pro" / "pace_competitions" / "pace2025_hs_agent" / "dataset") in command_text


def test_dry_run_writes_stage_status_and_manifest(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--provider",
            "glm-52",
            "--stage",
            "main21_agent",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert run(args) == 0

    status_path = tmp_path / "glm-52" / "runner_status" / "main21_agent.json"
    manifest_path = tmp_path / "provider_sweep_manifest.json"
    assert status_path.exists()
    assert manifest_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert status["status"] == "dry_run"
    assert status["provider"] == "glm-52"
    assert manifest["status"] == "dry_run"
    assert manifest["results"][0]["stage"] == "main21_agent"
