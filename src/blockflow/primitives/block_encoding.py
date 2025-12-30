from __future__ import annotations
from dataclasses import dataclass
import math
import numbers
from typing import Optional
import numpy as np

from .linear_operator import LinearOperator, NumpyMatrixOperator, DiagonalOperator, PermutationOperator
from .resources import ResourceEstimate
from .capabilities import Capabilities
from .success import SuccessModel
from .recipe import CircuitRecipe
from ..compile.circuit import Circuit
from ..compile.export_qasm import QasmFlavor, to_openqasm
from ..compile.optimizers import OptimizationOptions, optimize_circuit
from ..compile.synthesis import (
    LCUStrategy,
    can_synthesize_block_encoding_circuit,
    can_synthesize_unitary_circuit,
    required_ancillas_for_block_encoding,
    synthesize_block_encoding_circuit,
    synthesize_unitary_circuit,
)

def _validate_real(name: str, value: float, *, positive: bool = False, non_negative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_dimension(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class BlockEncoding:
    """
    Represents an (approximate) block encoding of A with normalization alpha.

    Convention
    semantic_apply returns A acting on the system state vector, not scaled by 1/alpha.
    alpha is tracked explicitly so algorithms can account for it.
    """
    op: LinearOperator
    alpha: float
    resources: ResourceEstimate
    success: SuccessModel = SuccessModel()
    capabilities: Capabilities = Capabilities()
    recipe: Optional[CircuitRecipe] = None
    epsilon: float = 0.0
    synthesis_strategy: Optional[LCUStrategy] = None

    def __post_init__(self) -> None:
        _validate_real("alpha", self.alpha, positive=True)
        _validate_real("epsilon", self.epsilon, non_negative=True)
        if self.synthesis_strategy is not None and self.synthesis_strategy not in ("prep_select", "sparse"):
            raise ValueError("synthesis_strategy must be 'prep_select' or 'sparse'")
        rows, cols = self._op_shape()
        if rows != cols:
            raise ValueError("BlockEncoding operator must be square")

    @classmethod
    def from_diagonal(
        cls,
        diagonal: np.ndarray,
        *,
        alpha: Optional[float] = None,
        resources: ResourceEstimate,
        success: SuccessModel = SuccessModel(),
        capabilities: Capabilities = Capabilities(),
        recipe: Optional[CircuitRecipe] = None,
        epsilon: float = 0.0,
    ) -> "BlockEncoding":
        diag = np.asarray(diagonal)
        if diag.ndim != 1:
            raise ValueError("diagonal must be a 1D array")
        max_entry = float(np.max(np.abs(diag))) if diag.size > 0 else 0.0
        if alpha is None:
            if math.isclose(max_entry, 0.0):
                raise ValueError("diagonal must be nonzero to infer alpha")
            alpha = max_entry
        return cls(
            op=DiagonalOperator(diag),
            alpha=alpha,
            resources=resources,
            success=success,
            capabilities=capabilities,
            recipe=recipe,
            epsilon=epsilon,
        )

    @classmethod
    def from_permutation(
        cls,
        permutation: np.ndarray,
        *,
        alpha: Optional[float] = None,
        resources: ResourceEstimate,
        success: SuccessModel = SuccessModel(),
        capabilities: Capabilities = Capabilities(),
        recipe: Optional[CircuitRecipe] = None,
        epsilon: float = 0.0,
    ) -> "BlockEncoding":
        if alpha is None:
            alpha = 1.0
        return cls(
            op=PermutationOperator(permutation),
            alpha=alpha,
            resources=resources,
            success=success,
            capabilities=capabilities,
            recipe=recipe,
            epsilon=epsilon,
        )

    def semantic_apply(self, vec: np.ndarray) -> np.ndarray:
        vec_arr = np.asarray(vec)
        rows, cols = self._op_shape()
        self._validate_vector(vec_arr, cols, "semantic_apply")
        out = np.asarray(self.op.apply(vec_arr))
        self._validate_vector(out, rows, "semantic_apply output")
        return out

    def semantic_apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        if not self.capabilities.supports_adjoint:
            raise ValueError("Adjoint not supported by this block encoding")
        vec_arr = np.asarray(vec)
        rows, cols = self._op_shape()
        self._validate_vector(vec_arr, rows, "semantic_apply_adjoint")
        out = np.asarray(self.op.apply_adjoint(vec_arr))
        self._validate_vector(out, cols, "semantic_apply_adjoint output")
        return out

    def can_export_circuit(self) -> bool:
        if not self.capabilities.supports_circuit_recipe:
            return False
        if self.recipe is not None:
            return True
        return self._can_synthesize_from_matrix()

    def build_circuit(self, *, optimize: bool = True, opts: Optional[OptimizationOptions] = None) -> Circuit:
        if self.recipe is None:
            if not self.capabilities.supports_circuit_recipe:
                raise ValueError("No circuit recipe attached to this block encoding")
            try:
                circ = self._build_circuit_from_matrix()
            except ValueError as exc:
                raise ValueError(f"No circuit recipe attached and matrix synthesis failed: {exc}") from exc
            if optimize:
                circ = optimize_circuit(circ, opts)
            return circ
        if not self.capabilities.supports_circuit_recipe:
            raise ValueError("Circuit export not supported; set capabilities.supports_circuit_recipe=True")
        wires = self.recipe.required_wires()
        required_qubits = wires.system_qubits + wires.ancilla_clean + wires.ancilla_dirty
        circ = self.recipe.build(optimize=optimize)
        if circ.num_qubits != required_qubits:
            raise ValueError(
                "Circuit qubit count does not match recipe WireSpec "
                f"(expected {required_qubits}, got {circ.num_qubits})"
            )
        if self.resources.ancilla_qubits_clean < wires.ancilla_clean:
            raise ValueError("Resource estimate has fewer clean ancillas than recipe requires")
        if self.resources.ancilla_qubits_dirty < wires.ancilla_dirty:
            raise ValueError("Resource estimate has fewer dirty ancillas than recipe requires")
        if optimize and opts is not None:
            circ = optimize_circuit(circ, opts)
        return circ

    def _can_synthesize_from_matrix(self) -> bool:
        if not isinstance(self.op, NumpyMatrixOperator):
            return False
        mat = self.op.mat
        strategy = self.synthesis_strategy or "prep_select"
        if can_synthesize_block_encoding_circuit(mat, alpha=self.alpha, strategy=strategy):
            try:
                required = required_ancillas_for_block_encoding(mat, alpha=self.alpha, strategy=strategy)
            except ValueError:
                return False
            if self.resources.ancilla_qubits_clean < required:
                return False
            return True
        return can_synthesize_unitary_circuit(mat / self.alpha)

    def _build_circuit_from_matrix(self) -> Circuit:
        if not isinstance(self.op, NumpyMatrixOperator):
            raise ValueError("Matrix synthesis requires NumpyMatrixOperator")
        mat = self.op.mat
        strategy = self.synthesis_strategy or "prep_select"
        if can_synthesize_block_encoding_circuit(mat, alpha=self.alpha, strategy=strategy):
            required = required_ancillas_for_block_encoding(mat, alpha=self.alpha, strategy=strategy)
            if self.resources.ancilla_qubits_clean >= required:
                return synthesize_block_encoding_circuit(mat, alpha=self.alpha, strategy=strategy)
            raise ValueError("Resource estimate has fewer clean ancillas than synthesis requires")
        mat = mat / self.alpha
        if can_synthesize_unitary_circuit(mat):
            return synthesize_unitary_circuit(mat)
        raise ValueError("Matrix synthesis supports 2x2 unitaries or Pauli-LCU block encodings")

    def export_openqasm(
        self,
        *,
        flavor: QasmFlavor = "qasm3",
        optimize: bool = True,
        opts: Optional[OptimizationOptions] = None,
    ) -> str:
        circ = self.build_circuit(optimize=optimize, opts=opts)
        return to_openqasm(circ, flavor=flavor)

    def _op_shape(self) -> tuple[int, int]:
        shape = self.op.shape
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError("Operator shape must be a length-2 tuple")
        rows = _validate_dimension("op.shape[0]", shape[0])
        cols = _validate_dimension("op.shape[1]", shape[1])
        return rows, cols

    @staticmethod
    def _validate_vector(vec: np.ndarray, expected_dim: int, label: str) -> None:
        if vec.ndim != 1:
            raise ValueError(f"{label} expects a 1D vector")
        if vec.shape[0] != expected_dim:
            raise ValueError(f"{label} has dimension {vec.shape[0]}, expected {expected_dim}")
