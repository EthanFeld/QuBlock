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
        scratch = getattr(state, "_scratch", None)
        for iteration in range(int(self.repeat)):
            if iteration == 0:
                # Infer the operator's output dtype before reusing storage. Reusing a
                # real state buffer for a complex result silently discards data on
                # NumPy and fails on stricter backends.
                out = self.be.semantic_apply(state.data)
                scratch = state.data
                state.data = out
            else:
                if (
                    scratch is None
                    or scratch.shape != state.data.shape
                    or scratch.dtype != state.data.dtype
                ):
                    scratch = backend.empty_like(state.data)
                out = self.be.semantic_apply(state.data, out=scratch)
                if out is scratch:
                    state.data, scratch = out, state.data
                else:
                    state.data = out
            report.include_use(
                success_prob=self.be.success.success_prob,
                anc_clean=self.be.resources.ancilla_qubits_clean,
                anc_dirty=self.be.resources.ancilla_qubits_dirty,
                resources=self.be.resources,
            )
        state._scratch = scratch


@dataclass
class ApplyBlockEncodingQuantumStep:
    """
    Apply the projected quantum block A/alpha using classical linear algebra.

    When postselect=True, retain and normalize the successful ancilla branch.
    The report tracks its state-dependent quantum success probability.
    """

    be: BlockEncoding
    repeat: int = 1
    adjoint: bool = False
    postselect: bool = True
    require_verified: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, numbers.Integral):
            raise TypeError("repeat must be a positive int")
        if self.repeat <= 0:
            raise ValueError("repeat must be a positive int")
        if self.adjoint and not self.be.capabilities.supports_adjoint:
            raise ValueError("Adjoint not supported by this block encoding")

    def run_semantic(self, state: StateVector, report: RunReport) -> None:
        status = self.be.validate_quantum_constraints(require_verified=self.require_verified)
        dimension = int(state.data.shape[0])
        if dimension <= 0 or dimension & (dimension - 1) != 0:
            raise ValueError("Quantum block encoding requires a power-of-two system dimension")
        for _ in range(int(self.repeat)):
            input_norm = backend.to_scalar(backend.linalg_norm(state.data))
            if input_norm == 0.0:
                raise ValueError("Cannot apply quantum block encoding to zero state")
            if self.adjoint:
                projected = self.be.semantic_apply_adjoint(state.data) / float(self.be.alpha)
            else:
                projected = self.be.semantic_apply(state.data) / float(self.be.alpha)
            projected_norm = backend.to_scalar(backend.linalg_norm(projected))
            branch_prob = (projected_norm / input_norm) ** 2
            if branch_prob > 1.0 + 1e-9:
                raise ValueError(
                    "Projected block produced probability > 1; alpha does not satisfy "
                    "block-encoding constraints"
                )
            branch_prob = min(max(branch_prob, 0.0), 1.0)
            success_prob = branch_prob * self.be.success.success_prob
            report.include_use(
                success_prob=success_prob,
                anc_clean=self.be.resources.ancilla_qubits_clean,
                anc_dirty=self.be.resources.ancilla_qubits_dirty,
                alpha=self.be.alpha,
                epsilon=self.be.epsilon,
                resources=self.be.resources,
                postselected=self.postselect,
                constraint_verified=status.valid is True,
            )
            if self.postselect:
                if projected_norm == 0.0:
                    raise ValueError("Cannot postselect a zero-probability block-encoding branch")
                projected = projected / projected_norm
            state.data = projected


@dataclass
class Program:
    steps: List[Step] = field(default_factory=list)

    def append(self, step: Step) -> None:
        self.steps.append(step)
