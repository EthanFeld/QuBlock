from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ResourceEstimate:
    """
    Lightweight resource summary for a circuit or primitive.

    Ancilla counts represent peak simultaneous usage, not totals.
    Gate and depth counts are additive for sequential composition.
    """
    ancilla_qubits_clean: int = 0
    ancilla_qubits_dirty: int = 0
    depth: int = 0
    one_qubit_gates: int = 0
    two_qubit_gates: int = 0
    t_count: int = 0
    measurements: int = 0

    def combine(self, other: "ResourceEstimate") -> "ResourceEstimate":
        """
        Sequentially compose two resource estimates.
        """
        return ResourceEstimate(
            ancilla_qubits_clean=max(self.ancilla_qubits_clean, other.ancilla_qubits_clean),
            ancilla_qubits_dirty=max(self.ancilla_qubits_dirty, other.ancilla_qubits_dirty),
            depth=self.depth + other.depth,
            one_qubit_gates=self.one_qubit_gates + other.one_qubit_gates,
            two_qubit_gates=self.two_qubit_gates + other.two_qubit_gates,
            t_count=self.t_count + other.t_count,
            measurements=self.measurements + other.measurements,
        )

    def scaled_by(self, factor: int) -> "ResourceEstimate":
        if factor < 0:
            raise ValueError("scale factor must be non-negative")
        return ResourceEstimate(
            ancilla_qubits_clean=self.ancilla_qubits_clean,
            ancilla_qubits_dirty=self.ancilla_qubits_dirty,
            depth=self.depth * factor,
            one_qubit_gates=self.one_qubit_gates * factor,
            two_qubit_gates=self.two_qubit_gates * factor,
            t_count=self.t_count * factor,
            measurements=self.measurements * factor,
        )
