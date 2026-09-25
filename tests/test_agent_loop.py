from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dasbench.agents.agent_loop as agent_loop
from dasbench.agents.agent_loop import _parse_action, run_agent_synthesis_loop
from dasbench.data import BenchmarkSpec, generate_dataset
from dasbench.eval.reporting import generate_benchmark_report


class _StubConfig:
    """Minimal stand-in for a ChatAPIConfig; the loop only needs public_dict()."""

    def public_dict(self) -> dict[str, object]:
        return {"provider": "stub", "model": "stub-model"}


# A scripted "agent" that writes a valid MIS candidate, checks it, and finishes.
_HYPOTHESIS = json.dumps({"diversity_key": "trivial_empty", "rule": "return an empty independent set"})
_ANALYZE = "def analyze(train_instances, manifest=None):\n    return {'train_count': len(train_instances)}\n"
_SOLUTION = "def solve(instance, analysis=None, manifest=None):\n    return []\n"

_SCRIPTED_REPLIES = [
    f"I will state a hypothesis.\nACTION: write_file hypothesis.json\n```json\n{_HYPOTHESIS}\n```",
    f"Now analyze.py.\nACTION: write_file analyze.py\n```python\n{_ANALYZE}```",
    f"Now solution.py.\nACTION: write_file solution.py\n```python\n{_SOLUTION}```",
    "Let me verify.\nACTION: run_check",
    "Looks good.\nACTION: finish",
]


class AgentLoopTests(unittest.TestCase):
    def test_parse_action_variants(self) -> None:
        write = _parse_action("thinking\nACTION: write_file solution.py\n```python\nx = 1\n```")
        self.assertEqual((write.verb, write.arg), ("write_file", "solution.py"))
        self.assertIn("x = 1", write.payload)

        check = _parse_action("ACTION: run_check")
        self.assertEqual(check.verb, "run_check")
        self.assertIsNone(check.payload)

        self.assertIsNone(_parse_action("no action at all"))

    def test_write_file_rejects_bad_payloads(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-agent-write-"))
        # Invalid python is rejected and not written.
        bad_py = _parse_action("ACTION: write_file analyze.py\n```python\ndef analyze(:\n```")
        msg = agent_loop._handle_write_file(root, bad_py)
        self.assertIn("syntax error", msg.lower())
        self.assertFalse((root / "analyze.py").exists())

        # Invalid JSON hypothesis is rejected.
        bad_json = _parse_action("ACTION: write_file hypothesis.json\n```json\n{not json}\n```")
        msg = agent_loop._handle_write_file(root, bad_json)
        self.assertIn("json", msg.lower())
        self.assertFalse((root / "hypothesis.json").exists())

        # Unknown target is rejected.
        bad_target = _parse_action("ACTION: write_file secrets.txt\n```\nhi\n```")
        self.assertIn("unknown target", agent_loop._handle_write_file(root, bad_target).lower())

    def test_scripted_agent_produces_comparable_candidate(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-agent-loop-"))
        dataset_dir = root / "dataset"
        run_dir = root / "run"
        report_dir = root / "report"
        spec = BenchmarkSpec(
            problem="mis",
            family="clique_path_mix_v1",
            instance_params={"num_vertices": 12},
            split_sizes={"train": 4, "validation": 2, "test": 2},
        )
        generate_dataset(dataset_dir, spec)

        replies = iter(_SCRIPTED_REPLIES)

        def _fake_chat(config, messages):  # noqa: ANN001 - matches agent_loop._chat
            return next(replies)

        with patch.object(agent_loop, "load_chat_api_config", return_value=_StubConfig()), patch.object(
            agent_loop, "_chat", _fake_chat
        ):
            summary = run_agent_synthesis_loop(
                dataset_dir,
                run_dir,
                mode="single",
                iterations=1,
                beam_width=1,
            )

        # Same summary schema as the llm/template generators.
        self.assertEqual(summary["problem"], "mis")
        self.assertEqual(summary["generator"], "agent")
        best = summary["best_candidate"]
        self.assertIn("test", best)
        self.assertIn("train", best)
        self.assertIn("validation", best)
        self.assertEqual(best["hypothesis"]["diversity_key"], "trivial_empty")
        # The scripted candidate returns a (feasible) empty independent set.
        self.assertEqual(best["train"]["feasibility_rate"], 1.0)

        # Candidate directory holds the required contract files plus agent traces.
        candidate_dir = Path(best["candidate_dir"])
        for name in ("hypothesis.json", "analyze.py", "solution.py", "agent_transcript.json"):
            self.assertTrue((candidate_dir / name).exists(), name)

        # Downstream reporting consumes the agent run exactly like any other generator.
        report = generate_benchmark_report(
            dataset_dir=dataset_dir,
            agent_run_dir=run_dir,
            output_dir=report_dir,
            repeats=1,
            include_train=False,
        )
        self.assertTrue(Path(report["json_path"]).exists())
        payload = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
        self.assertIn("gurobi_timed", payload["split_reports"]["test"])

    def test_resume_reuses_saved_record(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-agent-resume-"))
        dataset_dir = root / "dataset"
        run_dir = root / "run"
        spec = BenchmarkSpec(
            problem="mis",
            family="clique_path_mix_v1",
            instance_params={"num_vertices": 12},
            split_sizes={"train": 4, "validation": 2, "test": 2},
        )
        generate_dataset(dataset_dir, spec)

        def _script():
            return iter(_SCRIPTED_REPLIES)

        first = _script()
        with patch.object(agent_loop, "load_chat_api_config", return_value=_StubConfig()), patch.object(
            agent_loop, "_chat", lambda config, messages: next(first)
        ):
            run_agent_synthesis_loop(dataset_dir, run_dir, mode="single", iterations=1, beam_width=1)

        # A second run must not call the model again (records are reused from disk).
        def _boom(config, messages):  # noqa: ANN001
            raise AssertionError("agent should not re-query the model on resume")

        with patch.object(agent_loop, "load_chat_api_config", return_value=_StubConfig()), patch.object(
            agent_loop, "_chat", _boom
        ):
            summary = run_agent_synthesis_loop(dataset_dir, run_dir, mode="single", iterations=1, beam_width=1)
        self.assertEqual(summary["generator"], "agent")


if __name__ == "__main__":
    unittest.main()
