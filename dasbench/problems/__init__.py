from __future__ import annotations

from dasbench.problems.base import ExactSolveResult, ProblemDefinition, ScoreResult, SolveOutcome
from dasbench.problems.coloring import PROBLEM as COLORING_PROBLEM
from dasbench.problems.mdkp import PROBLEM as MDKP_PROBLEM
from dasbench.problems.mds import PROBLEM as MDS_PROBLEM
from dasbench.problems.maxsat import PROBLEM as MAXSAT_PROBLEM
from dasbench.problems.mis import PROBLEM as MIS_PROBLEM
from dasbench.problems.packing_lp import PROBLEM as PACKING_LP_PROBLEM
from dasbench.problems.tsp import PROBLEM as TSP_PROBLEM

PROBLEMS: dict[str, ProblemDefinition] = {
    "coloring": COLORING_PROBLEM,
    "mdkp": MDKP_PROBLEM,
    "maxsat": MAXSAT_PROBLEM,
    "mis": MIS_PROBLEM,
    "mds": MDS_PROBLEM,
    "pace": MDS_PROBLEM,
    "packing_lp": PACKING_LP_PROBLEM,
    "tsp": TSP_PROBLEM,
}


def available_problem_names() -> list[str]:
    return sorted(PROBLEMS)


def get_problem_definition(name: str) -> ProblemDefinition:
    try:
        return PROBLEMS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown problem `{name}`. Available problems: {', '.join(available_problem_names())}"
        ) from exc


__all__ = [
    "ExactSolveResult",
    "ProblemDefinition",
    "PROBLEMS",
    "ScoreResult",
    "SolveOutcome",
    "available_problem_names",
    "get_problem_definition",
]
