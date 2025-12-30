from __future__ import annotations
from dataclasses import dataclass
import numbers
from typing import Protocol

from ..compile.circuit import Circuit
from ..compile.optimizers import OptimizationOptions, optimize_circuit

@dataclass(frozen=True)
class WireSpec:
    system_qubits: int
    ancilla_clean: int = 0
    ancilla_dirty: int = 0

    def __post_init__(self) -> None:
        for field_name in ("system_qubits", "ancilla_clean", "ancilla_dirty"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{field_name} must be a non-negative int")
            if value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")

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
