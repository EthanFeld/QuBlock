from __future__ import annotations

import numpy as np
import pytest

from blockflow.primitives.capabilities import Capabilities
from blockflow.primitives.linear_operator import NumpyMatrixOperator
from blockflow.primitives.resources import ResourceEstimate
from blockflow.primitives.success import SuccessModel
from blockflow.semantic.state import StateVector
from blockflow.semantic.tracking import RunReport


def test_resource_estimate_combine_and_scale() -> None:
    a = ResourceEstimate(ancilla_qubits_clean=1, depth=2, one_qubit_gates=3)
    b = ResourceEstimate(ancilla_qubits_dirty=2, depth=5, two_qubit_gates=4)
    combined = a.combine(b)
    assert combined.ancilla_qubits_clean == 1
    assert combined.ancilla_qubits_dirty == 2
    assert combined.depth == 7
    assert combined.one_qubit_gates == 3
    assert combined.two_qubit_gates == 4

    scaled = combined.scaled_by(2)
    assert scaled.depth == 14
    assert scaled.two_qubit_gates == 8

    with pytest.raises(ValueError, match="non-negative"):
        combined.scaled_by(-1)


def test_numpy_operator_apply_and_norm_bound() -> None:
    mat = np.array([[1.0, 0.0], [0.0, -1.0]])
    op = NumpyMatrixOperator(mat, _norm_bound=5.0)
    vec = np.array([1.0, 2.0])
    assert np.allclose(op.apply(vec), np.array([1.0, -2.0]))
    assert np.allclose(op.apply_adjoint(vec), np.array([1.0, -2.0]))
    assert op.norm_bound() == 5.0

    op2 = NumpyMatrixOperator(mat)
    assert op2.norm_bound() >= 1.0


def test_statevector_normalize() -> None:
    state = StateVector(np.array([3.0, 4.0]))
    state.normalize()
    assert np.allclose(state.data, np.array([0.6, 0.8]))

    zero_state = StateVector(np.array([0.0, 0.0]))
    zero_state.normalize()
    assert np.allclose(zero_state.data, np.array([0.0, 0.0]))


def test_runreport_tracking() -> None:
    report = RunReport()
    report.include_use(success_prob=0.5, anc_clean=1, anc_dirty=2)
    report.include_use(success_prob=0.25, anc_clean=0, anc_dirty=3)
    assert report.uses == 2
    assert report.cumulative_success_prob == 0.125
    assert report.ancilla_clean_peak == 1
    assert report.ancilla_dirty_peak == 3


def test_misc_dataclasses() -> None:
    caps = Capabilities()
    success = SuccessModel(success_prob=0.9, notes="demo")
    assert caps.supports_adjoint
    assert success.success_prob == 0.9
    assert success.notes == "demo"
