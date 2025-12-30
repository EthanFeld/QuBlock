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

    def __post_init__(self) -> None:
        mat = np.asarray(self.mat)
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D array")
        object.__setattr__(self, "mat", mat)

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


@dataclass(frozen=True)
class DiagonalOperator:
    diag: np.ndarray

    def __post_init__(self) -> None:
        diag = np.asarray(self.diag)
        if diag.ndim != 1:
            raise ValueError("diag must be a 1D array")
        if diag.shape[0] <= 0:
            raise ValueError("diag must be non-empty")
        object.__setattr__(self, "diag", diag)

    @property
    def shape(self) -> tuple[int, int]:
        dim = int(self.diag.shape[0])
        return (dim, dim)

    @property
    def dtype(self):
        return self.diag.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.diag.shape[0]:
            raise ValueError("Vector dimension does not match diagonal operator")
        return self.diag * vec

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.diag.shape[0]:
            raise ValueError("Vector dimension does not match diagonal operator")
        return self.diag.conj() * vec

    def norm_bound(self) -> float:
        return float(np.max(np.abs(self.diag)))


@dataclass(frozen=True)
class PermutationOperator:
    perm: np.ndarray
    _inverse: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        perm = np.asarray(self.perm)
        if perm.ndim != 1:
            raise ValueError("perm must be a 1D array")
        if perm.shape[0] <= 0:
            raise ValueError("perm must be non-empty")
        if not np.issubdtype(perm.dtype, np.integer):
            raise TypeError("perm must contain integers")
        perm_list = perm.astype(int, copy=False)
        n = perm_list.shape[0]
        if set(perm_list.tolist()) != set(range(n)):
            raise ValueError("perm must be a permutation of 0..n-1")
        inv = np.empty_like(perm_list)
        inv[perm_list] = np.arange(n)
        object.__setattr__(self, "perm", perm_list)
        object.__setattr__(self, "_inverse", inv)

    @property
    def shape(self) -> tuple[int, int]:
        dim = int(self.perm.shape[0])
        return (dim, dim)

    @property
    def dtype(self):
        return self.perm.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.perm.shape[0]:
            raise ValueError("Vector dimension does not match permutation operator")
        return vec[self.perm]

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.perm.shape[0]:
            raise ValueError("Vector dimension does not match permutation operator")
        return vec[self._inverse]

    def norm_bound(self) -> float:
        return 1.0
