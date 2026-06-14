from __future__ import annotations

import numpy as np
import pytest

from blockflow import (
    ApplyBlockEncodingQuantumStep,
    BlockEncoding,
    Capabilities,
    NumpyMatrixOperator,
    Program,
    ResourceEstimate,
    SemanticExecutor,
    StateVector,
    SuccessModel,
)


def test_quantum_step_applies_projected_block_without_postselection() -> None:
    mat = np.diag([2.0, 1.0])
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=2.0,
        resources=ResourceEstimate(ancilla_qubits_clean=3, depth=7, oracle_queries=2),
        epsilon=0.01,
    )
    init = StateVector(np.array([1.0, 1.0]) / np.sqrt(2.0))

    final, report = SemanticExecutor().run(
        Program([ApplyBlockEncodingQuantumStep(be, postselect=False)]),
        init,
    )

    expected = np.array([1.0, 0.5]) / np.sqrt(2.0)
    assert np.allclose(final.data, expected)
    assert np.isclose(report.cumulative_success_prob, np.vdot(expected, expected).real)
    assert report.normalization_product == 2.0
    assert report.cumulative_error_bound == 0.01
    assert report.conditional_error_bound_valid
    assert report.ancilla_clean_peak == 3
    assert report.resources.depth == 7
    assert report.resources.oracle_queries == 2
    assert report.postselections == 0
    assert report.constraint_checks == 1
    assert report.unverified_constraints == 0


def test_quantum_algorithm_composes_projected_blocks_and_constraints() -> None:
    a = np.diag([2.0, 1.0])
    b = np.array([[0.0, 3.0], [1.0, 0.0]])
    be_a = BlockEncoding(
        op=NumpyMatrixOperator(a),
        alpha=2.0,
        resources=ResourceEstimate(depth=4, oracle_queries=1),
    )
    be_b = BlockEncoding(
        op=NumpyMatrixOperator(b),
        alpha=3.0,
        resources=ResourceEstimate(depth=6, oracle_queries=2),
    )
    psi = np.array([0.6, 0.8])

    final, report = SemanticExecutor().run(
        Program(
            [
                ApplyBlockEncodingQuantumStep(be_a, postselect=False),
                ApplyBlockEncodingQuantumStep(be_b, postselect=False),
            ]
        ),
        StateVector(psi),
    )

    expected = (b / 3.0) @ (a / 2.0) @ psi
    assert np.allclose(final.data, expected)
    assert np.isclose(report.cumulative_success_prob, np.vdot(expected, expected).real)
    assert report.normalization_product == 6.0
    assert report.resources.depth == 10
    assert report.resources.oracle_queries == 3
    assert report.uses == 2


def test_quantum_step_postselects_and_tracks_state_dependent_success() -> None:
    be = BlockEncoding(
        op=NumpyMatrixOperator(np.diag([1.0, 0.0])),
        alpha=1.0,
        resources=ResourceEstimate(postselections=1),
        success=SuccessModel(success_prob=0.5),
    )
    init = StateVector(np.array([0.5, np.sqrt(0.75)]))

    final, report = SemanticExecutor().run(
        Program([ApplyBlockEncodingQuantumStep(be)]),
        init,
    )

    assert np.allclose(final.data, np.array([1.0, 0.0]))
    assert np.isclose(report.cumulative_success_prob, 0.125)
    assert report.postselections == 1
    assert report.resources.postselections == 1
    assert np.isclose(report.expected_trials, 8.0)
    assert report.trials_for_confidence(0.9) == 18


def test_quantum_step_adjoint_and_repeat_accumulate_constraints() -> None:
    mat = np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=complex)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(depth=4),
        capabilities=Capabilities(supports_adjoint=True),
        epsilon=0.02,
    )

    final, report = SemanticExecutor().run(
        Program([ApplyBlockEncodingQuantumStep(be, repeat=2, adjoint=True)]),
        StateVector(np.array([1.0, 0.0], dtype=complex)),
    )

    assert np.allclose(final.data, np.array([-1.0j, 0.0]))
    assert report.uses == 2
    assert report.resources.depth == 8
    assert report.cumulative_error_bound == 0.04
    assert not report.conditional_error_bound_valid
    assert report.normalization_product == 1.0


def test_quantum_step_rejects_invalid_known_block_encoding() -> None:
    be = BlockEncoding(
        op=NumpyMatrixOperator(np.diag([2.0, 1.0])),
        alpha=1.0,
        resources=ResourceEstimate(),
    )

    with pytest.raises(ValueError, match="alpha must bound operator norm"):
        SemanticExecutor().run(
            Program([ApplyBlockEncodingQuantumStep(be)]),
            StateVector(np.array([1.0, 0.0])),
        )


def test_quantum_step_can_require_verified_operator_constraint() -> None:
    class MatrixFreeIdentity:
        shape = (2, 2)
        dtype = np.dtype(float)

        def apply(self, vec: np.ndarray) -> np.ndarray:
            return vec.copy()

        def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
            return vec.copy()

        def norm_bound(self) -> float:
            return 2.0

    be = BlockEncoding(op=MatrixFreeIdentity(), alpha=1.0, resources=ResourceEstimate())
    status = be.constraint_status()
    assert status.valid is None

    with pytest.raises(ValueError, match="Unverified block encoding constraint"):
        SemanticExecutor().run(
            Program([ApplyBlockEncodingQuantumStep(be, require_verified=True)]),
            StateVector(np.array([1.0, 0.0])),
        )

    _, report = SemanticExecutor().run(
        Program([ApplyBlockEncodingQuantumStep(be, require_verified=False)]),
        StateVector(np.array([1.0, 0.0])),
    )
    assert report.unverified_constraints == 1

    class CertifiedMatrixFreeIdentity(MatrixFreeIdentity):
        def norm_bound(self) -> float:
            return 1.0

    certified = BlockEncoding(
        op=CertifiedMatrixFreeIdentity(),
        alpha=1.0,
        resources=ResourceEstimate(),
    )
    _, certified_report = SemanticExecutor().run(
        Program([ApplyBlockEncodingQuantumStep(certified, require_verified=True)]),
        StateVector(np.array([1.0, 0.0])),
    )
    assert certified_report.constraints_verified


def test_quantum_step_rejects_zero_probability_postselection() -> None:
    be = BlockEncoding(
        op=NumpyMatrixOperator(np.diag([1.0, 0.0])),
        alpha=1.0,
        resources=ResourceEstimate(),
    )

    with pytest.raises(ValueError, match="zero-probability"):
        SemanticExecutor().run(
            Program([ApplyBlockEncodingQuantumStep(be)]),
            StateVector(np.array([0.0, 1.0])),
        )


def test_quantum_step_requires_power_of_two_system_dimension() -> None:
    be = BlockEncoding(
        op=NumpyMatrixOperator(np.eye(3)),
        alpha=1.0,
        resources=ResourceEstimate(),
    )

    with pytest.raises(ValueError, match="power-of-two"):
        SemanticExecutor().run(
            Program([ApplyBlockEncodingQuantumStep(be)]),
            StateVector(np.array([1.0, 0.0, 0.0])),
        )


def test_report_retry_estimates_and_validation() -> None:
    _, report = SemanticExecutor().run(Program(), StateVector(np.array([1.0, 0.0])))
    assert report.constraints_verified
    assert report.expected_trials == 1.0
    assert report.trials_for_confidence(0.99) == 1

    report.cumulative_success_prob = 0.0
    assert np.isinf(report.expected_trials)
    assert np.isinf(report.trials_for_confidence(0.9))
    with pytest.raises(TypeError, match="confidence"):
        report.trials_for_confidence("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence"):
        report.trials_for_confidence(1.0)
