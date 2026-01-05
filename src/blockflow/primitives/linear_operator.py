from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable
import numpy as np

from .. import backend

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
    mat: Any
    _norm_bound: Optional[float] = None

    def __post_init__(self) -> None:
        mat = backend.asarray(self.mat)
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
        return backend.matmul(self.mat, vec)

    def apply_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        return backend.matmul(self.mat, vec, out=out)

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        return backend.matmul(self.mat.conj().T, vec)

    def apply_adjoint_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        return backend.matmul(self.mat.conj().T, vec, out=out)

    def norm_bound(self) -> float:
        if self._norm_bound is not None:
            return float(self._norm_bound)
        # Conservative default. Users should override for big cases.
        try:
            return backend.to_scalar(backend.linalg_norm(self.mat, ord=2))
        except Exception:
            return float(np.linalg.norm(backend.to_numpy(self.mat), ord=2))

    def to_numpy(self) -> np.ndarray:
        return backend.to_numpy(self.mat)


@dataclass(frozen=True)
class DiagonalOperator:
    diag: Any

    def __post_init__(self) -> None:
        diag = backend.asarray(self.diag)
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
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.diag.shape[0]:
            raise ValueError("Vector dimension does not match diagonal operator")
        return backend.multiply(self.diag, vec)

    def apply_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.diag.shape[0]:
            raise ValueError("Vector dimension does not match diagonal operator")
        return backend.multiply(self.diag, vec, out=out)

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.diag.shape[0]:
            raise ValueError("Vector dimension does not match diagonal operator")
        return backend.multiply(self.diag.conj(), vec)

    def apply_adjoint_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.diag.shape[0]:
            raise ValueError("Vector dimension does not match diagonal operator")
        return backend.multiply(self.diag.conj(), vec, out=out)

    def norm_bound(self) -> float:
        return backend.to_scalar(backend.amax(backend.abs(self.diag)))


@dataclass(frozen=True)
class PermutationOperator:
    perm: Any
    _inverse: Optional[Any] = None

    def __post_init__(self) -> None:
        perm_obj = self.perm
        if hasattr(perm_obj, "is_floating_point") and callable(perm_obj.is_floating_point):
            if perm_obj.is_floating_point() or (
                hasattr(perm_obj, "is_complex") and callable(perm_obj.is_complex) and perm_obj.is_complex()
            ):
                raise TypeError("perm must contain integers")
            try:
                perm_cpu = np.asarray(perm_obj)
            except Exception:
                perm_cpu = backend.to_numpy(backend.asarray(perm_obj, dtype="int64"))
        else:
            perm_cpu = np.asarray(perm_obj)
        if perm_cpu.ndim != 1:
            raise ValueError("perm must be a 1D array")
        if perm_cpu.shape[0] <= 0:
            raise ValueError("perm must be non-empty")
        if not np.issubdtype(perm_cpu.dtype, np.integer):
            raise TypeError("perm must contain integers")
        perm_list = perm_cpu.astype(int, copy=False)
        n = perm_list.shape[0]
        if set(perm_list.tolist()) != set(range(n)):
            raise ValueError("perm must be a permutation of 0..n-1")
        inv = np.empty_like(perm_list)
        inv[perm_list] = np.arange(n)
        perm_arr = backend.asarray_like(perm_list, perm_obj, dtype="int64")
        inv_arr = backend.asarray_like(inv, perm_arr, dtype="int64")
        object.__setattr__(self, "perm", perm_arr)
        object.__setattr__(self, "_inverse", inv_arr)

    @property
    def shape(self) -> tuple[int, int]:
        dim = int(self.perm.shape[0])
        return (dim, dim)

    @property
    def dtype(self):
        return self.perm.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.perm.shape[0]:
            raise ValueError("Vector dimension does not match permutation operator")
        return backend.take(vec, self.perm)

    def apply_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.perm.shape[0]:
            raise ValueError("Vector dimension does not match permutation operator")
        return backend.take(vec, self.perm, out=out)

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.perm.shape[0]:
            raise ValueError("Vector dimension does not match permutation operator")
        return backend.take(vec, self._inverse)

    def apply_adjoint_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        vec = backend.asarray(vec)
        if vec.ndim != 1 or vec.shape[0] != self.perm.shape[0]:
            raise ValueError("Vector dimension does not match permutation operator")
        return backend.take(vec, self._inverse, out=out)

    def norm_bound(self) -> float:
        return 1.0


@dataclass(frozen=True)
class SparseMatrixOperator:
    mat: Any
    _norm_bound: Optional[float] = None

    def __post_init__(self) -> None:
        if not hasattr(self.mat, "shape"):
            raise ValueError("mat must define a shape")
        shape = getattr(self.mat, "shape")
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("mat must be a 2D array")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.mat.shape[0]), int(self.mat.shape[1]))

    @property
    def dtype(self):
        return self.mat.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return self.mat @ vec

    def apply_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        result = self.mat @ vec
        out[...] = result
        return out

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        return self.mat.conj().T @ vec

    def apply_adjoint_into(self, vec: np.ndarray, out: np.ndarray) -> np.ndarray:
        result = self.mat.conj().T @ vec
        out[...] = result
        return out

    def norm_bound(self) -> float:
        if self._norm_bound is not None:
            return float(self._norm_bound)
        if hasattr(self.mat, "norm"):
            try:
                return float(self.mat.norm())
            except Exception:
                pass
        if hasattr(self.mat, "toarray"):
            dense = self.mat.toarray()
            return float(np.linalg.norm(dense, ord=2))
        return float(np.linalg.norm(np.asarray(self.mat), ord=2))

    def to_numpy(self) -> np.ndarray:
        if hasattr(self.mat, "toarray"):
            return np.asarray(self.mat.toarray())
        return np.asarray(self.mat)
