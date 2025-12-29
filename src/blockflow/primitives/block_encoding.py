from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .linear_operator import LinearOperator
from .resources import ResourceEstimate
from .capabilities import Capabilities
from .success import SuccessModel
from .recipe import CircuitRecipe
from ..compile.circuit import Circuit
from ..compile.export_qasm import QasmFlavor, to_openqasm
from ..compile.optimizers import OptimizationOptions, optimize_circuit

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

    def semantic_apply(self, vec: np.ndarray) -> np.ndarray:
        return self.op.apply(vec)

    def semantic_apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        if not self.capabilities.supports_adjoint:
            raise ValueError("Adjoint not supported by this block encoding")
        return self.op.apply_adjoint(vec)

    def can_export_circuit(self) -> bool:
        return self.recipe is not None and self.capabilities.supports_circuit_recipe

    def build_circuit(self, *, optimize: bool = True, opts: Optional[OptimizationOptions] = None) -> Circuit:
        if self.recipe is None:
            raise ValueError("No circuit recipe attached to this block encoding")
        if not self.capabilities.supports_circuit_recipe:
            raise ValueError("Circuit export not supported; set capabilities.supports_circuit_recipe=True")
        circ = self.recipe.build(optimize=optimize)
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
