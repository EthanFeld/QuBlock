from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional

@dataclass(frozen=True)
class Gate:
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    controls: Tuple[int, ...] = ()

@dataclass
class Circuit:
    num_qubits: int
    gates: List[Gate] = field(default_factory=list)
    name: str = "blockflow"

    def add(
        self,
        name: str,
        qubits: list[int],
        params: Optional[list[float]] = None,
        controls: Optional[list[int]] = None,
    ) -> None:
        self._validate_gate(name=name, qubits=qubits, params=params, controls=controls)
        self.gates.append(
            Gate(
                name=name,
                qubits=tuple(qubits),
                params=tuple(params or ()),
                controls=tuple(controls or ()),
            )
        )

    def copy(self) -> "Circuit":
        return Circuit(num_qubits=self.num_qubits, gates=list(self.gates), name=self.name)

    def add_controlled(
        self,
        name: str,
        *,
        controls: list[int],
        targets: list[int],
        params: Optional[list[float]] = None,
    ) -> None:
        self.add(name, targets, params=params, controls=controls)

    def _validate_gate(
        self,
        *,
        name: str,
        qubits: list[int],
        params: Optional[list[float]],
        controls: Optional[list[int]],
    ) -> None:
        if not qubits:
            raise ValueError("Gate requires at least one qubit")
        if controls is not None and not isinstance(controls, list):
            raise TypeError("controls must be a list of ints or None")
        all_qubits = list(controls or []) + list(qubits)
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"Qubit index out of range: {q}")
        for q in controls or []:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"Qubit index out of range: {q}")
        if len(set(all_qubits)) != len(all_qubits):
            raise ValueError("Gate controls/targets must be unique")
        if params is not None and not isinstance(params, list):
            raise TypeError("params must be a list of floats or None")
