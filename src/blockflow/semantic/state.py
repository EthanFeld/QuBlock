from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class StateVector:
    """
    State of only the designated system wires for semantic execution.
    """
    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 1:
            raise ValueError("StateVector data must be a 1D array")

    def normalize(self) -> None:
        nrm = np.linalg.norm(self.data)
        if nrm == 0:
            return
        self.data = self.data / nrm
