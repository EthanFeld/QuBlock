from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

from ..compile.circuit import Circuit
from ..compile.optimizers import OptimizationOptions, optimize_circuit

@dataclass(frozen=True)
class WireSpec:
    system_qubits: int
    ancilla_clean: int = 0
    ancilla_dirty: int = 0

class CircuitRecipe(Protocol):
    """
    Constructive recipe for a block encoding circuit.
    Implementations should build the unitary that realizes the block encoding
    with the declared ancillas, not just the effective map.
    """
    def required_wires(self) -> WireSpec: ...
    def build(self, *, optimize: bool = True) -> Circuit:
        """
        Return a backend agnostic circuit object, defined in compile.circuit.
        """
        ...

@dataclass(frozen=True)
class StaticCircuitRecipe:
    """
    Simple recipe that returns a fixed circuit.
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
