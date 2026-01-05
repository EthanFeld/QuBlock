from __future__ import annotations

import numpy as np
import pytest

from blockflow import backend


def test_backend_numpy_ops(monkeypatch) -> None:
    monkeypatch.setenv("BLOCKFLOW_BACKEND", "numpy")
    monkeypatch.delenv("BLOCKFLOW_DTYPE", raising=False)

    vec = backend.asarray([1.0, 2.0])
    out = backend.empty_like(vec)
    res = backend.matmul(np.eye(2), vec, out=out)
    assert res is out
    assert np.allclose(res, np.array([1.0, 2.0]))

    mul = backend.multiply(vec, np.array([2.0, 3.0]))
    assert np.allclose(mul, np.array([2.0, 6.0]))

    taken = backend.take(vec, np.array([1, 0], dtype=int))
    assert np.allclose(taken, np.array([2.0, 1.0]))

    norm = backend.to_scalar(backend.linalg_norm(vec))
    assert np.isclose(norm, np.linalg.norm(np.array([1.0, 2.0])))

    copied = backend.copy(vec)
    assert copied is not vec
    assert np.allclose(copied, vec)


def test_backend_dtype_env(monkeypatch) -> None:
    monkeypatch.setenv("BLOCKFLOW_BACKEND", "numpy")
    monkeypatch.setenv("BLOCKFLOW_DTYPE", "complex64")
    vec = backend.asarray([1.0, 2.0])
    assert vec.dtype == np.complex64


def test_backend_auto_detects_numpy(monkeypatch) -> None:
    monkeypatch.setenv("BLOCKFLOW_BACKEND", "auto")
    arr = backend.asarray(np.array([1.0, 2.0]))
    assert isinstance(arr, np.ndarray)


def test_backend_internal_helpers(monkeypatch) -> None:
    assert backend._normalize_backend_name(None) == "numpy"
    assert backend._normalize_backend_name("") == "numpy"
    assert backend._normalize_backend_name("np") == "numpy"
    with pytest.raises(ValueError, match="Unknown backend"):
        backend._normalize_backend_name("bogus")

    class FakeCupy:
        __module__ = "cupy.core"

    class FakeTorch:
        __module__ = "torch.nn"

    assert backend._backend_from_object(FakeCupy()) == "cupy"
    assert backend._backend_from_object(FakeTorch()) == "torch"
    assert backend._backend_from_object(object()) is None

    assert backend._resolve_dtype_name(None) is None
    assert backend._resolve_dtype_name("") is None
    assert backend._resolve_dtype_name("C64") == "complex64"
    with pytest.raises(ValueError, match="Unknown dtype"):
        backend._resolve_dtype_name("bad")

    class DummyXP:
        pass

    with pytest.raises(ValueError, match="not supported"):
        backend._resolve_dtype(DummyXP(), "complex64")

    monkeypatch.setenv("BLOCKFLOW_DTYPE", "float32")
    with pytest.raises(ValueError, match="not supported"):
        backend._resolve_dtype(DummyXP(), None)

    assert backend._resolve_dtype(np, "") is None

    monkeypatch.setenv("BLOCKFLOW_DTYPE", "float32")
    assert backend._dtype_from_env() == "float32"

    monkeypatch.setenv("BLOCKFLOW_DTYPE", "bad")
    with pytest.raises(ValueError, match="Unknown dtype"):
        backend._dtype_from_env()


def test_backend_numpy_fallback_paths() -> None:
    class MinimalXP:
        linalg = np.linalg

        @staticmethod
        def asarray(x, dtype=None):
            return np.asarray(x, dtype=dtype)

        @staticmethod
        def empty_like(x, dtype=None):
            return np.empty_like(x, dtype=dtype)

        @staticmethod
        def matmul(a, b, out=None):
            raise TypeError("no out support")

        @staticmethod
        def multiply(a, b, out=None):
            raise TypeError("no out support")

        @staticmethod
        def take(a, indices, out=None):
            raise TypeError("no out support")

        @staticmethod
        def abs(x):
            return np.abs(x)

        @staticmethod
        def max(x):
            return np.max(x)

    backend_impl = backend._NumpyBackend(MinimalXP(), "numpy")
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    vec = np.array([1.0, 0.5])
    out = np.empty_like(vec)
    res = backend_impl.matmul(mat, vec, out=out)
    assert res is out
    assert np.allclose(res, mat @ vec)

    out_mul = np.empty_like(vec)
    res_mul = backend_impl.multiply(vec, np.array([2.0, 3.0]), out=out_mul)
    assert res_mul is out_mul
    assert np.allclose(res_mul, np.array([2.0, 1.5]))

    out_take = np.empty_like(vec)
    res_take = backend_impl.take(vec, np.array([1, 0]), out=out_take)
    assert res_take is out_take
    assert np.allclose(res_take, np.array([0.5, 1.0]))

    class ScalarLike:
        def __float__(self):
            raise TypeError("no float")

        def item(self):
            return 7.0

    assert backend_impl.to_scalar(ScalarLike()) == 7.0


def test_backend_invalid_backend_name(monkeypatch) -> None:
    monkeypatch.setenv("BLOCKFLOW_BACKEND", "bogus")
    with pytest.raises(ValueError, match="Unknown backend"):
        backend.asarray([1.0])


def test_backend_invalid_dtype_env(monkeypatch) -> None:
    monkeypatch.setenv("BLOCKFLOW_BACKEND", "numpy")
    monkeypatch.setenv("BLOCKFLOW_DTYPE", "bad")
    with pytest.raises(ValueError, match="Unknown dtype"):
        backend.asarray([1.0])
