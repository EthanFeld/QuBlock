from __future__ import annotations
from dataclasses import dataclass, field
import numbers
from typing import List, Protocol

from ..primitives.block_encoding import BlockEncoding
from ..semantic.state import StateVector
from ..semantic.tracking import RunReport
from .. import backend

class Step(Protocol):
    def run_semantic(self, state: StateVector, report: RunReport) -> None: ...

@dataclass
class ApplyBlockEncodingStep:
    be: BlockEncoding
    repeat: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, numbers.Integral):
            raise TypeError("repeat must be a positive int")
        if self.repeat <= 0:
            raise ValueError("repeat must be a positive int")

    def run_semantic(self, state: StateVector, report: RunReport) -> None:
        for _ in range(int(self.repeat)):
            scratch = getattr(state, "_scratch", None)
            if scratch is None or scratch.shape != state.data.shape or scratch.dtype != state.data.dtype:
                scratch = backend.empty_like(state.data)
            out = self.be.semantic_apply(state.data, out=scratch)
            if out is scratch:
                state.data, scratch = out, state.data
            else:
                state.data = out
            setattr(state, "_scratch", scratch)
            report.include_use(
                success_prob=self.be.success.success_prob,
                anc_clean=self.be.resources.ancilla_qubits_clean,
                anc_dirty=self.be.resources.ancilla_qubits_dirty,
            )

@dataclass
class Program:
    steps: List[Step] = field(default_factory=list)

    def append(self, step: Step) -> None:
        self.steps.append(step)
