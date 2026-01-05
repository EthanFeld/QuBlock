from __future__ import annotations
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..programs.program import Program
from .state import StateVector
from .tracking import RunReport
from .. import backend

_COPY_STATE_ENV = "BLOCKFLOW_COPY_STATE"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    return default

@dataclass
class SemanticExecutor:
    """
    Executes a Program purely semantically on the system statevector.
    """
    def run(
        self,
        program: "Program",
        init: StateVector,
        *,
        renormalize_each_step: bool = False,
        copy_state: Optional[bool] = None,
    ) -> tuple[StateVector, RunReport]:
        if copy_state is None:
            copy_state = _env_flag(_COPY_STATE_ENV, default=True)
        if copy_state:
            data = backend.copy(init.data)
            state = StateVector(data)
        else:
            state = init
        report = RunReport()
        for step in program.steps:
            step.run_semantic(state, report)
            if renormalize_each_step:
                state.normalize()
        return state, report
