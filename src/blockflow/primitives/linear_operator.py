from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Optional
import numpy as np

@runtime_checkable
class LinearOperator(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def dtype(self): ...
    def apply(self, vec: np.ndarray) -> np.ndarray: ...
    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray: ...
    def norm_bound(self) -> float: ...

@dataclass(frozen=True)
class NumpyMatrixOperator:
    mat: np.ndarray
    _norm_bound: Optional[float] = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.mat.shape[0]), int(self.mat.shape[1]))

    @property
    def dtype(self):
        return self.mat.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return self.mat @ vec

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        return self.mat.conj().T @ vec

    def norm_bound(self) -> float:
        if self._norm_bound is not None:
            return float(self._norm_bound)
        # Conservative default. Users should override for big cases.
        return float(np.linalg.norm(self.mat, ord=2))
