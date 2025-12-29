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
