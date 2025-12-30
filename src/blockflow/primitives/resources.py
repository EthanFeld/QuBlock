from __future__ import annotations
from dataclasses import dataclass
import numbers

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
    qram_queries: int = 0
    oracle_queries: int = 0
    classical_queries: int = 0
    postselections: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "ancilla_qubits_clean",
            "ancilla_qubits_dirty",
            "depth",
            "one_qubit_gates",
            "two_qubit_gates",
            "t_count",
            "measurements",
            "qram_queries",
            "oracle_queries",
            "classical_queries",
            "postselections",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{field_name} must be a non-negative int")
            if value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")

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
            qram_queries=self.qram_queries + other.qram_queries,
            oracle_queries=self.oracle_queries + other.oracle_queries,
            classical_queries=self.classical_queries + other.classical_queries,
            postselections=self.postselections + other.postselections,
        )

    def scaled_by(self, factor: int) -> "ResourceEstimate":
        if isinstance(factor, bool) or not isinstance(factor, numbers.Integral):
            raise TypeError("scale factor must be a non-negative int")
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
            qram_queries=self.qram_queries * factor,
            oracle_queries=self.oracle_queries * factor,
            classical_queries=self.classical_queries * factor,
            postselections=self.postselections * factor,
        )
