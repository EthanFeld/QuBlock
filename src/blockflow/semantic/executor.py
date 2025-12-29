from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..programs.program import Program
from .state import StateVector
from .tracking import RunReport

@dataclass
class SemanticExecutor:
    """
    Executes a Program purely semantically on the system statevector.
    """
    def run(self, program: "Program", init: StateVector, *, renormalize_each_step: bool = False) -> tuple[StateVector, RunReport]:
        state = StateVector(init.data.copy())
        report = RunReport()
        for step in program.steps:
            step.run_semantic(state, report)
            if renormalize_each_step:
                state.normalize()
        return state, report
