from __future__ import annotations

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from blockflow import (
    ApplyBlockEncodingStep,
    BlockEncoding,
    Capabilities,
    Circuit,
    NumpyMatrixOperator,
    Program,
    ResourceEstimate,
    SemanticExecutor,
    StateVector,
    StaticCircuitRecipe,
    WireSpec,
)


def _assert_state_global_phase(lhs: np.ndarray, rhs: np.ndarray, *, atol: float = 1e-7) -> None:
    denom = np.vdot(rhs, rhs)
    if abs(denom) < 1e-12:
        assert np.allclose(lhs, rhs, atol=atol)
        return
    phase = np.vdot(rhs, lhs) / denom
    if abs(phase) > 0:
        phase = phase / abs(phase)
    assert np.allclose(lhs, phase * rhs, atol=atol)


def _circuit_to_qiskit(circ: Circuit) -> QuantumCircuit:
    qc = QuantumCircuit(circ.num_qubits)
    for gate in circ.gates:
        name = gate.name.lower()
        controls = list(gate.controls)
        targets = list(gate.qubits)
        if len(targets) != 1:
            raise ValueError(f"Unsupported gate arity for Qiskit export: {gate}")
        target = targets[0]
        params = list(gate.params)

        if not controls:
            if name == "x":
                qc.x(target)
            elif name == "y":
                qc.y(target)
            elif name == "z":
                qc.z(target)
            elif name == "ry":
                qc.ry(params[0], target)
            elif name == "rz":
                qc.rz(params[0], target)
            else:
                raise ValueError(f"Unsupported gate for Qiskit export: {gate}")
            continue

        if len(controls) == 1:
            control = controls[0]
            if name == "x":
                qc.cx(control, target)
            elif name == "y":
                if hasattr(qc, "cy"):
                    qc.cy(control, target)
                else:
                    qc.sdg(target)
                    qc.cx(control, target)
                    qc.s(target)
            elif name == "z":
                if hasattr(qc, "cz"):
                    qc.cz(control, target)
                else:
                    qc.h(target)
                    qc.cx(control, target)
                    qc.h(target)
            elif name == "ry":
                qc.cry(params[0], control, target)
            elif name == "rz":
                qc.crz(params[0], control, target)
            else:
                raise ValueError(f"Unsupported controlled gate for Qiskit export: {gate}")
            continue

        if len(controls) == 2:
            if name == "x":
                qc.mcx(controls, target)
            elif name == "y":
                qc.sdg(target)
                qc.mcx(controls, target)
                qc.s(target)
            elif name == "z":
                qc.h(target)
                qc.mcx(controls, target)
                qc.h(target)
            elif name == "ry":
                if hasattr(qc, "mcry"):
                    qc.mcry(params[0], controls, target)
                else:
                    raise ValueError("Multi-controlled ry not supported in this Qiskit version")
            elif name == "rz":
                if hasattr(qc, "mcrz"):
                    qc.mcrz(params[0], controls, target)
                else:
                    raise ValueError("Multi-controlled rz not supported in this Qiskit version")
            else:
                raise ValueError(f"Unsupported multi-controlled gate for Qiskit export: {gate}")
            continue

        if len(controls) > 2:
            if name == "x":
                qc.mcx(controls, target)
            elif name == "y":
                qc.sdg(target)
                qc.mcx(controls, target)
                qc.s(target)
            elif name == "z":
                qc.h(target)
                qc.mcx(controls, target)
                qc.h(target)
            elif name == "ry":
                if hasattr(qc, "mcry"):
                    qc.mcry(params[0], controls, target)
                else:
                    raise ValueError("Multi-controlled ry not supported in this Qiskit version")
            elif name == "rz":
                if hasattr(qc, "mcrz"):
                    qc.mcrz(params[0], controls, target)
                else:
                    raise ValueError("Multi-controlled rz not supported in this Qiskit version")
            else:
                raise ValueError(f"Unsupported multi-controlled gate for Qiskit export: {gate}")
            continue

        raise ValueError(f"Unsupported control count for Qiskit export: {gate}")

    return qc


def _build_unitary_lcu_block_encoding() -> BlockEncoding:
    a = 0.6
    b = 0.8
    A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    B = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    lcu_matrix = a * A + b * B
    theta = np.arctan2(a, b)

    circ = Circuit(num_qubits=1)
    circ.add("ry", [0], [-theta])
    circ.add("z", [0])
    circ.add("ry", [0], [theta])

    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    return BlockEncoding(
        op=NumpyMatrixOperator(lcu_matrix),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )


@pytest.mark.parametrize(
    "psi",
    [
        np.array([1.0, 0.0], dtype=complex),
        np.array([0.0, 1.0], dtype=complex),
    ],
)
def test_qiskit_matches_semantic_for_lcu_unitary(psi: np.ndarray) -> None:
    be = _build_unitary_lcu_block_encoding()
    program = Program([ApplyBlockEncodingStep(be)])
    state = StateVector(psi)
    final_state, _ = SemanticExecutor().run(program, state)

    qasm2 = be.export_openqasm(flavor="qasm2", optimize=False)
    qc = QuantumCircuit.from_qasm_str(qasm2)
    qiskit_out = Statevector(psi).evolve(qc).data

    assert np.allclose(final_state.data, qiskit_out)


def test_qiskit_matches_semantic_for_block_encoding_synthesis() -> None:
    mat = np.array([[0.3 + 0.1j, -0.2 + 0.05j], [0.1 - 0.07j, -0.4 + 0.02j]], dtype=complex)
    I = np.eye(2, dtype=complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    coeffs = np.array(
        [
            0.5 * np.trace(I @ mat),
            0.5 * np.trace(X @ mat),
            0.5 * np.trace(Y @ mat),
            0.5 * np.trace(Z @ mat),
        ],
        dtype=complex,
    )
    alpha = float(np.sum(np.abs(coeffs)))

    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=alpha,
        resources=ResourceEstimate(ancilla_qubits_clean=3),
        capabilities=Capabilities(supports_circuit_recipe=True),
    )

    psi = np.array([0.6 + 0.1j, 0.3 - 0.7j], dtype=complex)
    psi = psi / np.linalg.norm(psi)
    program = Program([ApplyBlockEncodingStep(be)])
    final_state, _ = SemanticExecutor().run(program, StateVector(psi))

    circ = be.build_circuit(optimize=False)
    qc = _circuit_to_qiskit(circ)
    full_state = np.zeros(2**circ.num_qubits, dtype=complex)
    full_state[0] = psi[0]
    full_state[1] = psi[1]
    qiskit_out = Statevector(full_state).evolve(qc).data

    ancillas = range(1, circ.num_qubits)
    keep = [i for i in range(len(qiskit_out)) if all(((i >> a) & 1) == 0 for a in ancillas)]
    projected = qiskit_out[keep]
    expected = final_state.data / alpha
    _assert_state_global_phase(projected, expected)
