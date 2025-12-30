from __future__ import annotations

import math
from typing import Iterable, Literal
import numpy as np

from .circuit import Circuit

LCUStrategy = Literal["prep_select", "sparse"]

_DEFAULT_ATOL = 1e-8
_PAULI_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_PAULI_LOOKUP = {"I": _PAULI_I, "X": _PAULI_X, "Y": _PAULI_Y, "Z": _PAULI_Z}


def is_unitary(mat: np.ndarray, *, atol: float = _DEFAULT_ATOL) -> bool:
    mat = np.asarray(mat, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    ident = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, ident, atol=atol)


def can_synthesize_unitary_circuit(mat: np.ndarray, *, atol: float = _DEFAULT_ATOL) -> bool:
    mat = np.asarray(mat, dtype=complex)
    if mat.shape != (2, 2):
        return False
    return is_unitary(mat, atol=atol)


def synthesize_unitary_circuit(mat: np.ndarray, *, atol: float = _DEFAULT_ATOL) -> Circuit:
    mat = np.asarray(mat, dtype=complex)
    if mat.shape != (2, 2):
        raise ValueError("Matrix synthesis only supports 2x2 unitaries")
    if not is_unitary(mat, atol=atol):
        raise ValueError("Matrix synthesis requires a unitary matrix")

    det = np.linalg.det(mat)
    phase = np.angle(det)
    su2 = mat * np.exp(-0.5j * phase)

    a = su2[0, 0]
    b = su2[0, 1]
    a_abs = abs(a)
    b_abs = abs(b)

    theta = 2.0 * math.atan2(b_abs, a_abs)

    if b_abs < atol:
        phi = 0.0
        lam = -2.0 * np.angle(a)
    elif a_abs < atol:
        phi = -2.0 * np.angle(-b)
        lam = 0.0
    else:
        bprime = -b
        phi = -(np.angle(a) + np.angle(bprime))
        lam = np.angle(bprime) - np.angle(a)

    circ = Circuit(num_qubits=1)
    _add_rotation(circ, "rz", _canonicalize_angle(float(lam)))
    _add_rotation(circ, "ry", _canonicalize_angle(float(theta)))
    _add_rotation(circ, "rz", _canonicalize_angle(float(phi)))
    return circ


def can_synthesize_block_encoding_circuit(
    mat: np.ndarray,
    *,
    alpha: float,
    strategy: LCUStrategy = "prep_select",
    atol: float = _DEFAULT_ATOL,
) -> bool:
    mat = np.asarray(mat, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    if not math.isfinite(alpha) or alpha <= 0:
        return False
    if strategy not in ("prep_select", "sparse"):
        return False
    try:
        _num_qubits_from_dim(mat.shape[0])
    except ValueError:
        return False
    terms = _pauli_decompose(mat, atol=atol)
    if not terms:
        return False
    coeffs = np.array([coeff for coeff, _ in terms], dtype=complex)
    norm = float(np.sum(np.abs(coeffs)))
    if norm <= 0:
        return False
    return math.isclose(alpha, norm, rel_tol=1e-7, abs_tol=atol)


def synthesize_block_encoding_circuit(
    mat: np.ndarray,
    *,
    alpha: float,
    strategy: LCUStrategy = "prep_select",
    atol: float = _DEFAULT_ATOL,
) -> Circuit:
    mat = np.asarray(mat, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Block encoding synthesis requires a square matrix")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Block encoding synthesis requires alpha > 0")
    if strategy not in ("prep_select", "sparse"):
        raise ValueError("Unknown LCU synthesis strategy")
    n = _num_qubits_from_dim(mat.shape[0])
    terms = _pauli_decompose(mat, atol=atol)
    if not terms:
        raise ValueError("Block encoding synthesis requires a nonzero matrix")
    coeffs = np.array([coeff for coeff, _ in terms], dtype=complex)
    norm = float(np.sum(np.abs(coeffs)))
    if norm <= 0:
        raise ValueError("Block encoding synthesis requires a nonzero matrix")
    if not math.isclose(alpha, norm, rel_tol=1e-7, abs_tol=atol):
        raise ValueError("Block encoding synthesis requires alpha == sum(|Pauli coeffs|)")

    labels = [labels for _, labels in terms]
    weights = np.abs(coeffs) / norm
    phases = np.angle(coeffs)

    if len(terms) == 1:
        circ = Circuit(num_qubits=n)
        _append_pauli_string(circ, labels[0], list(range(n)), controls=[])
        return circ

    if strategy == "prep_select":
        return _synthesize_lcu_prep_select(labels, weights, phases, n, atol=atol)
    return _synthesize_lcu_sparse(labels, weights, phases, n, atol=atol)


def _canonicalize_angle(angle: float) -> float:
    return math.remainder(angle, 2.0 * math.pi)


def _add_rotation(circ: Circuit, name: str, angle: float, *, tol: float = 1e-12) -> None:
    if math.isclose(angle, 0.0, abs_tol=tol):
        return
    circ.add(name, [0], [angle])

def required_ancillas_for_block_encoding(
    mat: np.ndarray,
    *,
    alpha: float,
    strategy: LCUStrategy = "prep_select",
    atol: float = _DEFAULT_ATOL,
) -> int:
    mat = np.asarray(mat, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Block encoding synthesis requires a square matrix")
    if strategy not in ("prep_select", "sparse"):
        raise ValueError("Unknown LCU synthesis strategy")
    _num_qubits_from_dim(mat.shape[0])
    terms = _pauli_decompose(mat, atol=atol)
    if not terms:
        raise ValueError("Block encoding synthesis requires a nonzero matrix")
    coeffs = np.array([coeff for coeff, _ in terms], dtype=complex)
    norm = float(np.sum(np.abs(coeffs)))
    if norm <= 0:
        raise ValueError("Block encoding synthesis requires a nonzero matrix")
    if not math.isclose(alpha, norm, rel_tol=1e-7, abs_tol=atol):
        raise ValueError("Block encoding synthesis requires alpha == sum(|Pauli coeffs|)")
    if len(terms) == 1:
        return 0
    if strategy == "sparse":
        return len(terms)
    ancillas = int(math.ceil(math.log2(len(terms))))
    if _needs_phase(coeffs, atol=atol):
        ancillas += 1
    return ancillas


def _num_qubits_from_dim(dim: int) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError("Matrix dimension must be a power-of-two int")
    if dim <= 0 or dim & (dim - 1) != 0:
        raise ValueError("Matrix dimension must be a power of two")
    return int(math.log2(dim))


def _iter_pauli_labels(n: int) -> Iterable[tuple[str, ...]]:
    if n == 0:
        yield ()
        return
    for prefix in _iter_pauli_labels(n - 1):
        for label in ("I", "X", "Y", "Z"):
            yield prefix + (label,)


def _pauli_matrix_from_labels(labels: tuple[str, ...]) -> np.ndarray:
    if not labels:
        return np.array([[1.0]], dtype=complex)
    mat = _PAULI_LOOKUP[labels[0]]
    for label in labels[1:]:
        mat = np.kron(_PAULI_LOOKUP[label], mat)
    return mat


def _pauli_decompose(mat: np.ndarray, *, atol: float) -> list[tuple[complex, tuple[str, ...]]]:
    n = _num_qubits_from_dim(int(mat.shape[0]))
    scale = 1.0 / (2**n)
    terms: list[tuple[complex, tuple[str, ...]]] = []
    for labels in _iter_pauli_labels(n):
        pauli = _pauli_matrix_from_labels(labels)
        coeff = scale * np.trace(pauli @ mat)
        if abs(coeff) > atol:
            terms.append((coeff, labels))
    return terms


def _needs_phase(coeffs: np.ndarray, *, atol: float) -> bool:
    for coeff in coeffs:
        if abs(coeff) <= atol:
            continue
        if abs(_canonicalize_angle(float(np.angle(coeff)))) > atol:
            return True
    return False


def _synthesize_lcu_prep_select(
    labels: list[tuple[str, ...]],
    weights: np.ndarray,
    phases: np.ndarray,
    system_qubits: int,
    *,
    atol: float,
) -> Circuit:
    term_count = len(labels)
    index_width = int(math.ceil(math.log2(term_count)))
    needs_phase = any(abs(_canonicalize_angle(float(phase))) > atol for phase in phases)
    ancillas = index_width + (1 if needs_phase else 0)
    total_qubits = system_qubits + ancillas
    circ = Circuit(num_qubits=total_qubits)
    index_qubits = list(range(system_qubits, system_qubits + index_width))
    phase_qubit = system_qubits + index_width if needs_phase else None

    if index_width > 0:
        amplitudes = np.zeros(2**index_width, dtype=float)
        amplitudes[:term_count] = np.sqrt(weights)
        prep_ops = _build_binary_prep_ops(amplitudes, index_qubits, atol=atol)
        _apply_prep_ops(circ, prep_ops)
    else:
        prep_ops = []

    if needs_phase and phase_qubit is not None:
        circ.add("x", [phase_qubit])
        for term_index, phase in enumerate(phases):
            if abs(phase) <= atol:
                continue
            _apply_controlled_rz_for_index(
                circ,
                index_qubits,
                term_index,
                phase_qubit,
                2.0 * float(phase),
            )

    for term_index, term_labels in enumerate(labels):
        _apply_pauli_for_index(circ, index_qubits, term_index, term_labels, system_qubits)

    if index_width > 0:
        _apply_prep_ops_inverse(circ, prep_ops)
    if needs_phase and phase_qubit is not None:
        circ.add("x", [phase_qubit])
    return circ


def _synthesize_lcu_sparse(
    labels: list[tuple[str, ...]],
    weights: np.ndarray,
    phases: np.ndarray,
    system_qubits: int,
    *,
    atol: float,
) -> Circuit:
    term_count = len(labels)
    ancillas = term_count
    total_qubits = system_qubits + ancillas
    circ = Circuit(num_qubits=total_qubits)
    term_qubits = list(range(system_qubits, system_qubits + term_count))

    prep_ops = _build_unary_prep_ops(weights, term_qubits, atol=atol)
    _apply_prep_ops(circ, prep_ops)

    for term_index, phase in enumerate(phases):
        if abs(phase) <= atol:
            continue
        circ.add("rz", [term_qubits[term_index]], [float(phase)])

    for term_index, term_labels in enumerate(labels):
        controls = [term_qubits[term_index]]
        _append_pauli_string(circ, term_labels, list(range(system_qubits)), controls=controls)

    _apply_prep_ops_inverse(circ, prep_ops)
    return circ


def _build_binary_prep_ops(
    amplitudes: np.ndarray,
    index_qubits: list[int],
    *,
    atol: float,
) -> list[tuple[int, float, list[tuple[int, int]]]]:
    ops: list[tuple[int, float, list[tuple[int, int]]]] = []

    def rec(level: int, controls: list[tuple[int, int]], vec: np.ndarray) -> None:
        if level >= len(index_qubits):
            return
        half = vec.shape[0] // 2
        left = vec[:half]
        right = vec[half:]
        norm_left = float(np.linalg.norm(left))
        norm_right = float(np.linalg.norm(right))
        if norm_left <= atol and norm_right <= atol:
            return
        angle = 2.0 * math.atan2(norm_right, norm_left)
        if not math.isclose(angle, 0.0, abs_tol=atol):
            ops.append((index_qubits[level], angle, list(controls)))
        if norm_left > atol:
            rec(level + 1, controls + [(index_qubits[level], 0)], left)
        if norm_right > atol:
            rec(level + 1, controls + [(index_qubits[level], 1)], right)

    rec(0, [], amplitudes)
    return ops


def _build_unary_prep_ops(
    weights: np.ndarray,
    term_qubits: list[int],
    *,
    atol: float,
) -> list[tuple[int, float, list[tuple[int, int]]]]:
    ops: list[tuple[int, float, list[tuple[int, int]]]] = []
    remaining = 1.0
    for idx, weight in enumerate(weights):
        if weight <= atol:
            continue
        if remaining <= atol:
            break
        ratio = min(weight / remaining, 1.0)
        angle = 2.0 * math.asin(math.sqrt(ratio))
        controls = [(term_qubits[j], 0) for j in range(idx)]
        if not math.isclose(angle, 0.0, abs_tol=atol):
            ops.append((term_qubits[idx], angle, controls))
        remaining -= weight
    return ops


def _apply_prep_ops(circ: Circuit, ops: list[tuple[int, float, list[tuple[int, int]]]]) -> None:
    for target, angle, controls in ops:
        _apply_controlled_ry(circ, target, angle, controls)


def _apply_prep_ops_inverse(circ: Circuit, ops: list[tuple[int, float, list[tuple[int, int]]]]) -> None:
    for target, angle, controls in reversed(ops):
        _apply_controlled_ry(circ, target, -angle, controls)


def _apply_controlled_ry(
    circ: Circuit,
    target: int,
    angle: float,
    controls: list[tuple[int, int]],
) -> None:
    if math.isclose(angle, 0.0, abs_tol=1e-12):
        return
    flips = [q for q, bit in controls if bit == 0]
    for q in flips:
        circ.add("x", [q])
    ctrl_qubits = [q for q, _ in controls]
    if ctrl_qubits:
        circ.add("ry", [target], [angle], controls=ctrl_qubits)
    else:
        circ.add("ry", [target], [angle])
    for q in reversed(flips):
        circ.add("x", [q])


def _apply_controlled_rz_for_index(
    circ: Circuit,
    index_qubits: list[int],
    index: int,
    target: int,
    angle: float,
) -> None:
    if math.isclose(angle, 0.0, abs_tol=1e-12):
        return
    bits = _index_bits(index, len(index_qubits))
    controls = list(zip(index_qubits, bits))
    _apply_controlled_rz(circ, target, angle, controls)


def _apply_controlled_rz(
    circ: Circuit,
    target: int,
    angle: float,
    controls: list[tuple[int, int]],
) -> None:
    if math.isclose(angle, 0.0, abs_tol=1e-12):
        return
    flips = [q for q, bit in controls if bit == 0]
    for q in flips:
        circ.add("x", [q])
    ctrl_qubits = [q for q, _ in controls]
    if ctrl_qubits:
        circ.add("rz", [target], [angle], controls=ctrl_qubits)
    else:
        circ.add("rz", [target], [angle])
    for q in reversed(flips):
        circ.add("x", [q])


def _apply_pauli_for_index(
    circ: Circuit,
    index_qubits: list[int],
    index: int,
    labels: tuple[str, ...],
    system_qubits: int,
) -> None:
    if not index_qubits:
        _append_pauli_string(circ, labels, list(range(system_qubits)), controls=[])
        return
    bits = _index_bits(index, len(index_qubits))
    flips = [q for q, bit in zip(index_qubits, bits) if bit == 0]
    for q in flips:
        circ.add("x", [q])
    _append_pauli_string(circ, labels, list(range(system_qubits)), controls=index_qubits)
    for q in reversed(flips):
        circ.add("x", [q])


def _append_pauli_string(
    circ: Circuit,
    labels: tuple[str, ...],
    system_qubits: list[int],
    *,
    controls: list[int],
) -> None:
    for idx, label in enumerate(labels):
        if label == "I":
            continue
        name = label.lower()
        if controls:
            circ.add(name, [system_qubits[idx]], controls=controls)
        else:
            circ.add(name, [system_qubits[idx]])


def _index_bits(index: int, width: int) -> list[int]:
    if width <= 0:
        return []
    return [(index >> (width - 1 - bit)) & 1 for bit in range(width)]
