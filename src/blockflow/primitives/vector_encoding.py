from __future__ import annotations
from dataclasses import dataclass
import math
import numbers
from typing import Optional, Protocol
import numpy as np

from .capabilities import Capabilities
from .resources import ResourceEstimate
from .success import SuccessModel
from .recipe import WireSpec
from ..compile.circuit import Circuit
from ..compile.export_qasm import QasmFlavor, to_openqasm
from ..compile.optimizers import OptimizationOptions, optimize_circuit


def _validate_real(name: str, value: float, *, positive: bool = False, non_negative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be >= 0")


class StatePreparationRecipe(Protocol):
    """
    Recipe for preparing a vector-encoded state from |0>.
    """
    def required_wires(self) -> WireSpec: ...
    def build(self, *, optimize: bool = True) -> Circuit: ...


@dataclass(frozen=True)
class StaticStatePreparationRecipe:
    """
    Simple recipe that returns a fixed state-preparation circuit.
    """
    wires: WireSpec
    circuit: Circuit

    def required_wires(self) -> WireSpec:
        return self.wires

    def build(self, *, optimize: bool = True) -> Circuit:
        built = self.circuit.copy()
        if optimize:
            built = optimize_circuit(built, OptimizationOptions())
        return built


@dataclass(frozen=True)
class VectorEncoding:
    """
    Represents a vector-encoding with normalization alpha.
    """
    vec: np.ndarray
    alpha: float
    resources: ResourceEstimate
    success: SuccessModel = SuccessModel()
    capabilities: Capabilities = Capabilities()
    recipe: Optional[StatePreparationRecipe] = None
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        _validate_real("alpha", self.alpha, positive=True)
        _validate_real("epsilon", self.epsilon, non_negative=True)
        vec_arr = np.asarray(self.vec)
        if vec_arr.ndim != 1:
            raise ValueError("vec must be a 1D array")
        if vec_arr.shape[0] <= 0:
            raise ValueError("vec must be non-empty")
        object.__setattr__(self, "vec", vec_arr)

    @classmethod
    def from_vector(
        cls,
        vec: np.ndarray,
        *,
        resources: ResourceEstimate,
        success: SuccessModel = SuccessModel(),
        capabilities: Capabilities = Capabilities(),
        recipe: Optional[StatePreparationRecipe] = None,
        epsilon: float = 0.0,
    ) -> "VectorEncoding":
        vec_arr = np.asarray(vec)
        if vec_arr.ndim != 1:
            raise ValueError("vec must be a 1D array")
        norm = float(np.linalg.norm(vec_arr))
        if math.isclose(norm, 0.0):
            raise ValueError("vec must be nonzero")
        return cls(
            vec=vec_arr,
            alpha=norm,
            resources=resources,
            success=success,
            capabilities=capabilities,
            recipe=recipe,
            epsilon=epsilon,
        )

    @property
    def dimension(self) -> int:
        return int(self.vec.shape[0])

    def semantic_state(self) -> np.ndarray:
        vec_arr = np.asarray(self.vec)
        if vec_arr.ndim != 1:
            raise ValueError("vec must be a 1D array")
        return vec_arr / float(self.alpha)

    def can_export_circuit(self) -> bool:
        return self.capabilities.supports_circuit_recipe and self.recipe is not None

    def build_circuit(self, *, optimize: bool = True, opts: Optional[OptimizationOptions] = None) -> Circuit:
        if self.recipe is None:
            raise ValueError("No state preparation recipe attached to this vector encoding")
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

    def export_openqasm(
        self,
        *,
        flavor: QasmFlavor = "qasm3",
        optimize: bool = True,
        opts: Optional[OptimizationOptions] = None,
    ) -> str:
        circ = self.build_circuit(optimize=optimize, opts=opts)
        return to_openqasm(circ, flavor=flavor)
