from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Protocol
import numpy as np

from ..primitives.block_encoding import BlockEncoding
from ..semantic.state import StateVector
from ..semantic.tracking import RunReport

class Step(Protocol):
    def run_semantic(self, state: StateVector, report: RunReport) -> None: ...

@dataclass
class ApplyBlockEncodingStep:
    be: BlockEncoding

    def run_semantic(self, state: StateVector, report: RunReport) -> None:
        state.data = self.be.semantic_apply(state.data)
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
