from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

_BACKEND_ENV = "BLOCKFLOW_BACKEND"
_DTYPE_ENV = "BLOCKFLOW_DTYPE"
_DEVICE_ENV = "BLOCKFLOW_DEVICE"

_BACKEND_ALIASES = {
    "np": "numpy",
    "numpy": "numpy",
    "cp": "cupy",
    "cupy": "cupy",
    "torch": "torch",
    "pytorch": "torch",
    "auto": "auto",
}

_DTYPE_ALIASES = {
    "float32": "float32",
    "f32": "float32",
    "float64": "float64",
    "f64": "float64",
    "complex64": "complex64",
    "c64": "complex64",
    "complex128": "complex128",
    "c128": "complex128",
    "int32": "int32",
    "i32": "int32",
    "int64": "int64",
    "i64": "int64",
}


def _normalize_backend_name(value: Optional[str]) -> str:
    if value is None:
        return "numpy"
    name = value.strip().lower()
    if not name:
        return "numpy"
    if name in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[name]
    raise ValueError(f"Unknown backend '{value}' (use numpy, cupy, torch, or auto)")


def _backend_from_object(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    module = obj.__class__.__module__
    if module.startswith("cupy"):
        return "cupy"
    if module.startswith("torch"):
        return "torch"
    return None


def _resolve_dtype_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    name = value.strip().lower()
    if not name:
        return None
    if name in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[name]
    raise ValueError(f"Unknown dtype '{value}' (use float32/64 or complex64/128)")


def _dtype_from_env() -> Optional[str]:
    return _resolve_dtype_name(os.getenv(_DTYPE_ENV))


def _resolve_dtype(xp: Any, dtype: Any) -> Any:
    if dtype is None:
        dtype_name = _dtype_from_env()
        if dtype_name is None:
            return None
        if hasattr(xp, dtype_name):
            return getattr(xp, dtype_name)
        raise ValueError(f"Dtype '{dtype_name}' not supported by backend '{xp}'")
    if isinstance(dtype, str):
        dtype_name = _resolve_dtype_name(dtype)
        if dtype_name is None:
            return None
        if hasattr(xp, dtype_name):
            return getattr(xp, dtype_name)
        raise ValueError(f"Dtype '{dtype_name}' not supported by backend '{xp}'")
    return dtype


@dataclass(frozen=True)
class _NumpyBackend:
    xp: Any
    name: str

    def asarray(self, x: Any, *, dtype: Any = None) -> Any:
        resolved = _resolve_dtype(self.xp, dtype)
        return self.xp.asarray(x, dtype=resolved)

    def asarray_like(self, x: Any, like: Any, *, dtype: Any = None) -> Any:
        resolved = _resolve_dtype(self.xp, dtype)
        return self.xp.asarray(x, dtype=resolved)

    def empty_like(self, x: Any, *, dtype: Any = None) -> Any:
        resolved = _resolve_dtype(self.xp, dtype)
        return self.xp.empty_like(x, dtype=resolved)

    def copy(self, x: Any) -> Any:
        return x.copy()

    def matmul(self, a: Any, b: Any, *, out: Any = None) -> Any:
        try:
            return self.xp.matmul(a, b, out=out)
        except TypeError:
            result = a @ b
            if out is None:
                return result
            out[...] = result
            return out

    def multiply(self, a: Any, b: Any, *, out: Any = None) -> Any:
        try:
            return self.xp.multiply(a, b, out=out)
        except TypeError:
            result = a * b
            if out is None:
                return result
            out[...] = result
            return out

    def take(self, a: Any, indices: Any, *, out: Any = None) -> Any:
        try:
            return self.xp.take(a, indices, out=out)
        except TypeError:
            result = a[indices]
            if out is None:
                return result
            out[...] = result
            return out

    def linalg_norm(self, x: Any, ord: Any = None) -> Any:
        if ord is None:
            return self.xp.linalg.norm(x)
        return self.xp.linalg.norm(x, ord=ord)

    def to_scalar(self, x: Any) -> float:
        try:
            return float(x)
        except TypeError:
            return float(x.item())

    def abs(self, x: Any) -> Any:
        return self.xp.abs(x)

    def amax(self, x: Any) -> Any:
        return self.xp.max(x)

    def size(self, x: Any) -> int:
        return int(x.size)

    def to_numpy(self, x: Any) -> np.ndarray:
        if self.name == "cupy":  # pragma: no cover - optional backend
            return x.get()
        return np.asarray(x)


class _TorchBackend:  # pragma: no cover - optional backend
    name = "torch"

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def _resolve_dtype(self, dtype: Any) -> Any:
        if dtype is None:
            dtype_name = _dtype_from_env()
            if dtype_name is None:
                return None
            return getattr(self.torch, dtype_name)
        if isinstance(dtype, str):
            dtype_name = _resolve_dtype_name(dtype)
            if dtype_name is None:
                return None
            return getattr(self.torch, dtype_name)
        return dtype

    def _resolve_device(self, like: Any = None) -> Any:
        device_name = os.getenv(_DEVICE_ENV)
        if device_name:
            return self.torch.device(device_name)
        if like is not None and isinstance(like, self.torch.Tensor):
            return like.device
        return None

    def asarray(self, x: Any, *, dtype: Any = None) -> Any:
        tensor = x if isinstance(x, self.torch.Tensor) else self.torch.as_tensor(x)
        resolved = self._resolve_dtype(dtype)
        if resolved is not None:
            tensor = tensor.to(dtype=resolved)
        device = self._resolve_device(tensor)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    def asarray_like(self, x: Any, like: Any, *, dtype: Any = None) -> Any:
        tensor = x if isinstance(x, self.torch.Tensor) else self.torch.as_tensor(x)
        resolved = self._resolve_dtype(dtype)
        if resolved is not None:
            tensor = tensor.to(dtype=resolved)
        device = self._resolve_device(like)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    def empty_like(self, x: Any, *, dtype: Any = None) -> Any:
        resolved = self._resolve_dtype(dtype)
        device = self._resolve_device(x)
        return self.torch.empty_like(x, dtype=resolved, device=device)

    def copy(self, x: Any) -> Any:
        return x.clone()

    def matmul(self, a: Any, b: Any, *, out: Any = None) -> Any:
        try:
            return self.torch.matmul(a, b, out=out)
        except TypeError:
            result = a @ b
            if out is None:
                return result
            out.copy_(result)
            return out

    def multiply(self, a: Any, b: Any, *, out: Any = None) -> Any:
        try:
            return self.torch.mul(a, b, out=out)
        except TypeError:
            result = a * b
            if out is None:
                return result
            out.copy_(result)
            return out

    def take(self, a: Any, indices: Any, *, out: Any = None) -> Any:
        idx = indices
        if not isinstance(idx, self.torch.Tensor):
            idx = self.torch.as_tensor(idx, device=a.device)
        elif idx.device != a.device:
            idx = idx.to(device=a.device)
        try:
            return self.torch.take(a, idx, out=out)
        except TypeError:
            result = self.torch.take(a, idx)
            if out is None:
                return result
            out.copy_(result)
            return out

    def linalg_norm(self, x: Any, ord: Any = None) -> Any:
        if ord is None:
            return self.torch.linalg.norm(x)
        return self.torch.linalg.norm(x, ord=ord)

    def to_scalar(self, x: Any) -> float:
        try:
            return float(x)
        except TypeError:
            return float(x.item())

    def abs(self, x: Any) -> Any:
        return self.torch.abs(x)

    def amax(self, x: Any) -> Any:
        return self.torch.max(x)

    def size(self, x: Any) -> int:
        return int(x.numel())

    def to_numpy(self, x: Any) -> np.ndarray:
        return x.detach().cpu().numpy()


_BACKEND_CACHE: dict[str, Any] = {}


def _load_cupy_backend() -> _NumpyBackend:  # pragma: no cover - optional backend
    module = importlib.import_module("cupy")
    return _NumpyBackend(module, "cupy")


def _load_torch_backend() -> _TorchBackend:  # pragma: no cover - optional backend
    module = importlib.import_module("torch")
    return _TorchBackend(module)


def _get_backend(name: str) -> Any:
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    if name == "numpy":
        backend = _NumpyBackend(np, "numpy")
    elif name == "cupy":  # pragma: no cover - optional backend
        backend = _load_cupy_backend()
    elif name == "torch":  # pragma: no cover - optional backend
        backend = _load_torch_backend()
    else:
        raise ValueError(f"Unknown backend '{name}'")
    _BACKEND_CACHE[name] = backend
    return backend


def _select_backend_name(obj: Any = None) -> str:
    name = _normalize_backend_name(os.getenv(_BACKEND_ENV))
    if name == "auto":
        detected = _backend_from_object(obj)
        return detected or "numpy"
    return name


def backend_name(obj: Any = None) -> str:
    return _select_backend_name(obj)


def asarray(x: Any, *, dtype: Any = None) -> Any:
    backend = _get_backend(_select_backend_name(x))
    return backend.asarray(x, dtype=dtype)


def asarray_like(x: Any, like: Any, *, dtype: Any = None) -> Any:
    backend = _get_backend(_select_backend_name(like))
    return backend.asarray_like(x, like, dtype=dtype)


def empty_like(x: Any, *, dtype: Any = None) -> Any:
    backend = _get_backend(_select_backend_name(x))
    return backend.empty_like(x, dtype=dtype)


def copy(x: Any) -> Any:
    backend = _get_backend(_select_backend_name(x))
    return backend.copy(x)


def matmul(a: Any, b: Any, *, out: Any = None) -> Any:
    backend = _get_backend(_select_backend_name(a if a is not None else b))
    return backend.matmul(a, b, out=out)


def multiply(a: Any, b: Any, *, out: Any = None) -> Any:
    backend = _get_backend(_select_backend_name(a if a is not None else b))
    return backend.multiply(a, b, out=out)


def take(a: Any, indices: Any, *, out: Any = None) -> Any:
    backend = _get_backend(_select_backend_name(a))
    return backend.take(a, indices, out=out)


def linalg_norm(x: Any, ord: Any = None) -> Any:
    backend = _get_backend(_select_backend_name(x))
    return backend.linalg_norm(x, ord=ord)


def to_scalar(x: Any) -> float:
    backend = _get_backend(_select_backend_name(x))
    return backend.to_scalar(x)


def abs(x: Any) -> Any:
    backend = _get_backend(_select_backend_name(x))
    return backend.abs(x)


def amax(x: Any) -> Any:
    backend = _get_backend(_select_backend_name(x))
    return backend.amax(x)


def size(x: Any) -> int:
    backend = _get_backend(_select_backend_name(x))
    return backend.size(x)


def to_numpy(x: Any) -> np.ndarray:
    backend = _get_backend(_select_backend_name(x))
    return backend.to_numpy(x)
