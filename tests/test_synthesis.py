from __future__ import annotations

import numpy as np
import pytest

from blockflow.compile.synthesis import (
    can_synthesize_block_encoding_circuit,
    can_synthesize_unitary_circuit,
    is_unitary,
    required_ancillas_for_block_encoding,
    synthesize_block_encoding_circuit,
    synthesize_unitary_circuit,
)
from blockflow.compile.synthesis import (
    _apply_controlled_ry,
    _apply_controlled_rz,
    _apply_controlled_rz_for_index,
    _apply_pauli_for_index,
    _build_binary_prep_ops,
    _build_unary_prep_ops,
    _index_bits,
    _iter_pauli_labels,
    _needs_phase,
    _num_qubits_from_dim,
    _pauli_matrix_from_labels,
    _synthesize_lcu_prep_select,
    _synthesize_lcu_sparse,
)
from blockflow.compile.circuit import Circuit


def test_unitary_checks_and_errors() -> None:
    non_square = np.zeros((2, 3), dtype=complex)
    assert not is_unitary(non_square)
    assert not can_synthesize_unitary_circuit(non_square)

    non_unitary = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    assert not is_unitary(non_unitary)
    assert not can_synthesize_unitary_circuit(non_unitary)
    with pytest.raises(ValueError, match="unitary"):
        synthesize_unitary_circuit(non_unitary)

    with pytest.raises(ValueError, match="2x2"):
        synthesize_unitary_circuit(np.eye(3, dtype=complex))


def test_unitary_synthesis_branches() -> None:
    diag_unitary = np.array([[1.0j, 0.0], [0.0, -1.0j]], dtype=complex)
    circ_diag = synthesize_unitary_circuit(diag_unitary)
    assert circ_diag.num_qubits == 1

    x_unitary = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    circ_x = synthesize_unitary_circuit(x_unitary)
    assert any(g.name.lower() == "ry" for g in circ_x.gates)


def test_block_encoding_checks_and_errors() -> None:
    mat = np.eye(2, dtype=complex)
    assert can_synthesize_block_encoding_circuit(mat, alpha=1.0)
    assert not can_synthesize_block_encoding_circuit(mat, alpha=0.5)
    assert not can_synthesize_block_encoding_circuit(mat, alpha=-1.0)
    assert not can_synthesize_block_encoding_circuit(np.eye(3, dtype=complex), alpha=1.0)

    with pytest.raises(ValueError, match="power of two"):
        synthesize_block_encoding_circuit(np.eye(3, dtype=complex), alpha=1.0)
    with pytest.raises(ValueError, match="alpha > 0"):
        synthesize_block_encoding_circuit(mat, alpha=0.0)
    with pytest.raises(ValueError, match="nonzero matrix"):
        synthesize_block_encoding_circuit(np.zeros((2, 2), dtype=complex), alpha=1.0)
    with pytest.raises(ValueError, match="alpha == sum"):
        synthesize_block_encoding_circuit(mat, alpha=2.0)


def test_block_encoding_sparse_terms() -> None:
    mat_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    circ_x = synthesize_block_encoding_circuit(mat_x, alpha=1.0)
    assert circ_x.num_qubits == 1

    mat_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    circ_z = synthesize_block_encoding_circuit(mat_z, alpha=1.0)
    assert circ_z.num_qubits == 1


def test_block_encoding_strategy_validation() -> None:
    mat = np.eye(2, dtype=complex)
    assert not can_synthesize_block_encoding_circuit(mat, alpha=1.0, strategy="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown LCU synthesis strategy"):
        synthesize_block_encoding_circuit(mat, alpha=1.0, strategy="unknown")  # type: ignore[arg-type]


def test_required_ancillas_validation() -> None:
    mat = np.eye(2, dtype=complex)
    assert required_ancillas_for_block_encoding(mat, alpha=1.0) == 0
    with pytest.raises(ValueError, match="square matrix"):
        required_ancillas_for_block_encoding(np.zeros((2, 3), dtype=complex), alpha=1.0)
    with pytest.raises(ValueError, match="Unknown LCU synthesis strategy"):
        required_ancillas_for_block_encoding(mat, alpha=1.0, strategy="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha == sum"):
        required_ancillas_for_block_encoding(mat, alpha=2.0)


def test_required_ancillas_strategy_counts() -> None:
    labels = [("X", "I"), ("I", "Z"), ("Z", "X")]
    coeffs_real = np.array([0.25, 0.25, 0.5], dtype=complex)
    mat_real = sum(
        coeff * _pauli_matrix_from_labels(label)
        for coeff, label in zip(coeffs_real, labels, strict=True)
    )
    alpha_real = float(np.sum(np.abs(coeffs_real)))
    assert required_ancillas_for_block_encoding(mat_real, alpha=alpha_real, strategy="prep_select") == 2
    assert required_ancillas_for_block_encoding(mat_real, alpha=alpha_real, strategy="sparse") == 3

    coeffs_complex = np.array([0.25, 0.25j, 0.5], dtype=complex)
    mat_complex = sum(
        coeff * _pauli_matrix_from_labels(label)
        for coeff, label in zip(coeffs_complex, labels, strict=True)
    )
    alpha_complex = float(np.sum(np.abs(coeffs_complex)))
    assert required_ancillas_for_block_encoding(mat_complex, alpha=alpha_complex, strategy="prep_select") == 3


def test_block_encoding_edge_cases() -> None:
    non_square = np.zeros((2, 3), dtype=complex)
    assert not can_synthesize_block_encoding_circuit(non_square, alpha=1.0)
    with pytest.raises(ValueError, match="square matrix"):
        synthesize_block_encoding_circuit(non_square, alpha=1.0)

    zero_mat = np.zeros((2, 2), dtype=complex)
    assert not can_synthesize_block_encoding_circuit(zero_mat, alpha=1.0, atol=-1.0)
    with pytest.raises(ValueError, match="nonzero matrix"):
        synthesize_block_encoding_circuit(zero_mat, alpha=1.0, atol=-1.0)
    with pytest.raises(ValueError, match="nonzero matrix"):
        required_ancillas_for_block_encoding(zero_mat, alpha=1.0, atol=-1.0)


def test_internal_helpers_cover_branches() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        _num_qubits_from_dim(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="power of two"):
        _num_qubits_from_dim(3)

    assert list(_iter_pauli_labels(0)) == [()]
    assert np.allclose(_pauli_matrix_from_labels(()), np.array([[1.0]], dtype=complex))
    assert _index_bits(0, 0) == []
    assert _needs_phase(np.array([0.0, 1j]), atol=1e-12)
    assert not _needs_phase(np.array([0.0, 0.0]), atol=1e-12)

    amplitudes = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
    ops = _build_binary_prep_ops(amplitudes, [0, 1], atol=1e-12)
    assert ops
    assert _build_binary_prep_ops(np.zeros(4), [0, 1], atol=1e-12) == []

    unary_ops = _build_unary_prep_ops(np.array([1.0, 0.5, 0.0]), [0, 1, 2], atol=1e-12)
    assert unary_ops

    circ = Circuit(num_qubits=2)
    _apply_controlled_ry(circ, 0, 0.0, [])
    _apply_controlled_ry(circ, 0, 0.25, [(1, 0)])
    _apply_controlled_rz(circ, 0, 0.0, [])
    _apply_controlled_rz(circ, 0, 0.5, [(1, 1)])
    _apply_controlled_rz(circ, 0, 0.3, [])
    _apply_controlled_rz_for_index(circ, [1], 0, 0, 0.0)
    _apply_pauli_for_index(circ, [], 0, ("X",), 1)


def test_direct_lcu_builders_cover_branches() -> None:
    labels = [("X",)]
    weights = np.array([1.0])
    phases = np.array([0.25])
    circ = _synthesize_lcu_prep_select(labels, weights, phases, 1, atol=1e-12)
    assert circ.num_qubits == 2

    sparse_labels = [("X",), ("Z",)]
    sparse_weights = np.array([0.6, 0.4])
    sparse_phases = np.array([0.0, 0.0])
    circ_sparse = _synthesize_lcu_sparse(sparse_labels, sparse_weights, sparse_phases, 1, atol=1e-12)
    assert circ_sparse.num_qubits == 3
