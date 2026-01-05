from __future__ import annotations

import numpy as np
import pytest

from blockflow.primitives.capabilities import Capabilities
from blockflow import backend
from blockflow.primitives.linear_operator import NumpyMatrixOperator, SparseMatrixOperator
from blockflow.primitives.recipe import WireSpec
from blockflow.primitives.resources import ResourceEstimate
from blockflow.primitives.success import SuccessModel
from blockflow.semantic.state import StateVector
from blockflow.semantic.tracking import RunReport


def test_resource_estimate_combine_and_scale() -> None:
    a = ResourceEstimate(ancilla_qubits_clean=1, depth=2, one_qubit_gates=3, qram_queries=2)
    b = ResourceEstimate(ancilla_qubits_dirty=2, depth=5, two_qubit_gates=4, oracle_queries=3)
    combined = a.combine(b)
    assert combined.ancilla_qubits_clean == 1
    assert combined.ancilla_qubits_dirty == 2
    assert combined.depth == 7
    assert combined.one_qubit_gates == 3
    assert combined.two_qubit_gates == 4
    assert combined.qram_queries == 2
    assert combined.oracle_queries == 3

    scaled = combined.scaled_by(2)
    assert scaled.depth == 14
    assert scaled.two_qubit_gates == 8
    assert scaled.qram_queries == 4

    with pytest.raises(ValueError, match="non-negative"):
        combined.scaled_by(-1)

    with pytest.raises(ValueError, match="non-negative"):
        ResourceEstimate(ancilla_qubits_clean=-1)
    with pytest.raises(TypeError, match="non-negative"):
        ResourceEstimate(depth=1.5)
    with pytest.raises(TypeError, match="non-negative"):
        combined.scaled_by(1.5)


def test_numpy_operator_apply_and_norm_bound() -> None:
    mat = np.array([[1.0, 0.0], [0.0, -1.0]])
    op = NumpyMatrixOperator(mat, _norm_bound=5.0)
    vec = np.array([1.0, 2.0])
    assert np.allclose(op.apply(vec), np.array([1.0, -2.0]))
    assert np.allclose(op.apply_adjoint(vec), np.array([1.0, -2.0]))
    out = np.empty_like(vec)
    assert op.apply_into(vec, out) is out
    assert np.allclose(out, np.array([1.0, -2.0]))
    out_adj = np.empty_like(vec)
    assert op.apply_adjoint_into(vec, out_adj) is out_adj
    assert np.allclose(out_adj, np.array([1.0, -2.0]))
    assert op.norm_bound() == 5.0
    assert op.dtype == mat.dtype

    op2 = NumpyMatrixOperator(mat)
    assert op2.norm_bound() >= 1.0

    with pytest.raises(ValueError, match="2D"):
        NumpyMatrixOperator(np.array([1.0, 2.0]))


def test_numpy_operator_norm_bound_fallback(monkeypatch) -> None:
    mat = np.eye(2)
    op = NumpyMatrixOperator(mat)

    def raise_norm(_x, ord=None):
        raise TypeError("boom")

    monkeypatch.setattr(backend, "linalg_norm", raise_norm)
    assert np.isclose(op.norm_bound(), np.linalg.norm(mat, ord=2))


def test_sparse_operator_apply_and_norm_bound() -> None:
    mat = np.array([[1.0, 0.0], [0.0, 2.0]])
    op = SparseMatrixOperator(mat, _norm_bound=3.0)
    vec = np.array([1.0, 2.0])
    assert np.allclose(op.apply(vec), np.array([1.0, 4.0]))
    out = np.empty_like(vec)
    assert op.apply_into(vec, out) is out
    assert np.allclose(out, np.array([1.0, 4.0]))
    assert op.apply_adjoint(vec).shape == vec.shape
    out_adj = np.empty_like(vec)
    assert op.apply_adjoint_into(vec, out_adj) is out_adj
    assert op.norm_bound() == 3.0


def test_sparse_operator_norm_and_toarray_paths() -> None:
    class NormMat:
        def __init__(self):
            self.shape = (2, 2)
            self.dtype = np.float64

        def norm(self):
            return 7.0

        def __matmul__(self, other):
            return np.eye(2) @ other

        def conj(self):
            return self

        @property
        def T(self):
            return self

    op = SparseMatrixOperator(NormMat())
    assert op.norm_bound() == 7.0

    class ToArrayMat:
        def __init__(self):
            self.shape = (2, 2)
            self.dtype = np.float64

        def norm(self):
            raise ValueError("nope")

        def toarray(self):
            return np.array([[2.0, 0.0], [0.0, 1.0]])

        def __matmul__(self, other):
            return self.toarray() @ other

        def conj(self):
            return self

        @property
        def T(self):
            return self

    op_toarray = SparseMatrixOperator(ToArrayMat())
    assert np.isclose(op_toarray.norm_bound(), np.linalg.norm(op_toarray.to_numpy(), ord=2))
    assert isinstance(op_toarray.to_numpy(), np.ndarray)

    class SimpleMat:
        def __init__(self):
            self.shape = (1, 1)
            self.dtype = np.float64

        def __matmul__(self, other):
            return np.array([other[0]])

        def conj(self):
            return self

        @property
        def T(self):
            return self

    op_simple = SparseMatrixOperator(SimpleMat())
    assert isinstance(op_simple.to_numpy(), np.ndarray)


def test_sparse_operator_validation_errors() -> None:
    class NoShape:
        pass

    with pytest.raises(ValueError, match="shape"):
        SparseMatrixOperator(NoShape())

    class BadShape:
        shape = [2, 2]
        dtype = np.float64

    with pytest.raises(ValueError, match="2D"):
        SparseMatrixOperator(BadShape())


def test_statevector_normalize() -> None:
    state = StateVector(np.array([3.0, 4.0]))
    state.normalize()
    assert np.allclose(state.data, np.array([0.6, 0.8]))

    zero_state = StateVector(np.array([0.0, 0.0]))
    zero_state.normalize()
    assert np.allclose(zero_state.data, np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="1D"):
        StateVector(np.eye(2))


def test_statevector_normalize_fallback_path() -> None:
    state = StateVector(np.array([1, 1], dtype=int))
    state.normalize()
    assert np.allclose(state.data, np.array([0.70710678, 0.70710678]))


def test_runreport_tracking() -> None:
    report = RunReport()
    report.include_use(success_prob=0.5, anc_clean=1, anc_dirty=2)
    report.include_use(success_prob=0.25, anc_clean=0, anc_dirty=3)
    assert report.uses == 2
    assert report.cumulative_success_prob == 0.125
    assert report.ancilla_clean_peak == 1
    assert report.ancilla_dirty_peak == 3

    with pytest.raises(ValueError, match="between 0 and 1"):
        report.include_use(success_prob=1.5, anc_clean=0, anc_dirty=0)
    with pytest.raises(TypeError, match="between 0 and 1"):
        report.include_use(success_prob="bad", anc_clean=0, anc_dirty=0)
    with pytest.raises(ValueError, match="finite"):
        report.include_use(success_prob=float("inf"), anc_clean=0, anc_dirty=0)
    with pytest.raises(TypeError, match="non-negative"):
        report.include_use(success_prob=0.5, anc_clean=1.5, anc_dirty=0)
    with pytest.raises(ValueError, match="non-negative"):
        report.include_use(success_prob=0.5, anc_clean=-1, anc_dirty=0)


def test_misc_dataclasses() -> None:
    caps = Capabilities()
    success = SuccessModel(success_prob=0.9, notes="demo")
    assert caps.supports_adjoint
    assert success.success_prob == 0.9
    assert success.notes == "demo"

    with pytest.raises(ValueError, match="between 0 and 1"):
        SuccessModel(success_prob=-0.1)
    with pytest.raises(TypeError, match="between 0 and 1"):
        SuccessModel(success_prob="bad")
    with pytest.raises(ValueError, match="finite"):
        SuccessModel(success_prob=float("inf"))

    with pytest.raises(TypeError, match="non-negative"):
        WireSpec(system_qubits=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        WireSpec(system_qubits=-1)
