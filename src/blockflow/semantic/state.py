from __future__ import annotations
from dataclasses import dataclass

from .. import backend

@dataclass
class StateVector:
    """
    State of only the designated system wires for semantic execution.
    """
    data: object

    def __post_init__(self) -> None:
        self.data = backend.asarray(self.data)
        if self.data.ndim != 1:
            raise ValueError("StateVector data must be a 1D array")

    def normalize(self) -> None:
        nrm = backend.to_scalar(backend.linalg_norm(self.data))
        if nrm == 0:
            return
        try:
            self.data /= nrm
        except Exception:
            self.data = self.data / nrm
