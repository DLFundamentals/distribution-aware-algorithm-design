from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.agents.llm import (
    LLMPlan,
    _build_analyze_messages,
    _build_hypothesis_messages,
    _build_solution_messages,
    _select_survivors,
)
from dasbench.utils import candidate_manifest


class AgentPromptSanitizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "problem": "maxsat",
            "family": "latent_backdoor_mixture_v1",
            "description": "Paper-grade family with hidden structural regimes.",
            "metric_definition": {"primary": "normalized_quality"},
            "instance_schema_version": "maxsat.v1",
            "instance_params": {"num_variables": 10, "num_clauses": 18},
            "family_params": {"hidden_regimes": 3},
            "ground_truth_hidden_rule": {
                "summary": "Private latent backdoor rule.",
                "signals": ["regime"],
            },
        }
        self.train_summary = {
            "problem": "maxsat",
            "family": "latent_backdoor_mixture_v1",
            "num_instances": 4,
            "sample_instances": [],
        }

    def test_candidate_manifest_hides_family_metadata(self) -> None:
        exposed = candidate_manifest(self.manifest)
        self.assertEqual(exposed["problem"], "maxsat")
        self.assertIn("distribution_note", exposed)
        self.assertNotIn("family", exposed)
        self.assertNotIn("description", exposed)
        self.assertNotIn("family_params", exposed)
        self.assertNotIn("ground_truth_hidden_rule", exposed)

    def test_prompt_payloads_hide_family_metadata(self) -> None:
        plan = LLMPlan(iteration=0, slot=0, focus="test")
        hypothesis_messages = _build_hypothesis_messages(
            manifest=self.manifest,
            train_summary=self.train_summary,
            plan=plan,
            parent_record=None,
        )
        hypothesis_payload = json.loads(hypothesis_messages[1]["content"])
        self.assertNotIn("family", hypothesis_payload["manifest"])
        self.assertNotIn("description", hypothesis_payload["manifest"])
        self.assertNotIn("family_params", hypothesis_payload["manifest"])
        self.assertNotIn("ground_truth_hidden_rule", hypothesis_payload["manifest"])
        self.assertNotIn("family", hypothesis_payload["train_summary"])
        self.assertIn("hypothesis", hypothesis_payload["stage"])

        hypothesis = {
            "title": "Anchor rule",
            "rule_summary": "Variables follow an anchor.",
            "evidence_to_measure": ["literal polarity"],
            "solver_strategy": "Use the anchor.",
            "expected_failure_modes": ["noise"],
            "diversity_key": "anchor",
        }
        analyze_messages = _build_analyze_messages(
            manifest=self.manifest,
            train_summary=self.train_summary,
            plan=plan,
            parent_record=None,
            hypothesis=hypothesis,
        )
        analyze_payload = json.loads(analyze_messages[1]["content"])
        self.assertNotIn("family", analyze_payload["manifest"])
        self.assertNotIn("description", analyze_payload["manifest"])
        self.assertNotIn("ground_truth_hidden_rule", analyze_payload["manifest"])
        self.assertNotIn("family", analyze_payload["train_summary"])
        self.assertEqual(analyze_payload["current_hypothesis"]["diversity_key"], "anchor")
        self.assertIn("unknown structured distribution", analyze_payload["manifest"]["distribution_note"])

        solution_messages = _build_solution_messages(
            manifest=self.manifest,
            train_summary=self.train_summary,
            plan=plan,
            parent_record=None,
            analyze_py="def analyze(train_instances, manifest=None):\n    return {}\n",
            analysis_output={"signal": 1},
            hypothesis=hypothesis,
        )
        solution_payload = json.loads(solution_messages[1]["content"])
        self.assertNotIn("family", solution_payload["manifest"])
        self.assertNotIn("ground_truth_hidden_rule", solution_payload["manifest"])
        self.assertNotIn("family", solution_payload["train_summary_overview"])
        self.assertEqual(solution_payload["current_hypothesis"]["diversity_key"], "anchor")
        self.assertIn("unknown structured distribution", solution_payload["manifest"]["distribution_note"])

    def test_dfvs_prompts_use_arcs_schema(self) -> None:
        manifest = {
            "problem": "dfvs",
            "family": "pace2022_dfvs_heuristic_private",
            "metric_definition": {"primary": "normalized_quality"},
            "instance_schema_version": "dfvs.v1",
            "instance_params": {},
        }
        train_summary = {
            "problem": "dfvs",
            "family": "pace2022_dfvs_heuristic_private",
            "num_instances": 1,
            "runtime_instance_fields": {
                "arcs": "list of directed 0-based [tail, head] pairs",
            },
            "sample_instances": [{"arc_prefix": [[0, 1]]}],
        }
        plan = LLMPlan(iteration=0, slot=0, focus="test")

        analyze_payload = json.loads(
            _build_analyze_messages(
                manifest=manifest,
                train_summary=train_summary,
                plan=plan,
                parent_record=None,
                hypothesis=None,
            )[1]["content"]
        )
        solution_payload = json.loads(
            _build_solution_messages(
                manifest=manifest,
                train_summary=train_summary,
                plan=plan,
                parent_record=None,
                analyze_py="def analyze(train_instances, manifest=None):\n    return {}\n",
                analysis_output={},
                hypothesis=None,
            )[1]["content"]
        )

        analyze_constraints = "\n".join(analyze_payload["constraints"])
        solution_constraints = "\n".join(solution_payload["constraints"])
        self.assertIn("instance['arcs']", analyze_constraints)
        self.assertIn("instance['arcs']", solution_constraints)
        self.assertIn("do not read instance['edges']", solution_constraints)
        self.assertNotIn("Runtime graph instances expose their edge list as instance['edges']", solution_constraints)

    def test_candidate_runtime_manifest_is_sanitized(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-agent-sanitize-"))
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "analyze.py").write_text(
            "def analyze(train_instances, manifest=None):\n"
            "    return {'manifest': manifest}\n",
            encoding="utf-8",
        )
        (candidate_dir / "solution.py").write_text(
            "def solve(instance, analysis=None, manifest=None):\n"
            "    return manifest\n",
            encoding="utf-8",
        )

        analysis = run_analysis(
            candidate_dir,
            train_instances=[{"id": "train-0000"}],
            manifest=self.manifest,
        )
        assert analysis is not None
        self.assertNotIn("family", analysis["manifest"])
        self.assertNotIn("description", analysis["manifest"])
        self.assertNotIn("ground_truth_hidden_rule", analysis["manifest"])

        solver = build_solver(candidate_dir, analysis=analysis, manifest=self.manifest)
        result = solver({"id": "test-0000"})
        self.assertNotIn("family", result)
        self.assertNotIn("description", result)
        self.assertNotIn("ground_truth_hidden_rule", result)

    def test_hypothesis_beam_preserves_diversity_before_filling(self) -> None:
        def record(slug: str, quality: float, diversity_key: str) -> dict[str, object]:
            return {
                "slug": slug,
                "hypothesis": {"diversity_key": diversity_key},
                "selection": {
                    "mean_normalized_quality": quality,
                    "mean_optimality_rate": quality,
                    "mean_runtime_ms": 1.0,
                },
            }

        selected = _select_survivors(
            [
                record("a", 1.00, "same"),
                record("b", 0.99, "same"),
                record("c", 0.98, "other"),
            ],
            mode="beam",
            beam_width=2,
        )
        self.assertEqual([item["slug"] for item in selected], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
