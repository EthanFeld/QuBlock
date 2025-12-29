from __future__ import annotations

import numpy as np
import pytest

import blockflow
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


def test_semantic_execution_predicts_output() -> None:
    mat = np.array([[0, 1], [1, 0]], dtype=complex)
    be = BlockEncoding(op=NumpyMatrixOperator(mat), alpha=1.0, resources=ResourceEstimate())
    program = Program([ApplyBlockEncodingStep(be)])
    state = StateVector(np.array([1.0, 0.0], dtype=complex))
    final_state, report = SemanticExecutor().run(program, state)

    assert np.allclose(final_state.data, np.array([0.0, 1.0], dtype=complex))
    assert report.uses == 1
    assert report.cumulative_success_prob == 1.0


def test_semantic_apply_adjoint_respects_capabilities() -> None:
    mat = np.eye(2, dtype=complex)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        capabilities=Capabilities(supports_adjoint=False),
    )
    with pytest.raises(ValueError, match="Adjoint not supported"):
        be.semantic_apply_adjoint(np.array([1.0, 0.0], dtype=complex))


def test_circuit_export_requires_recipe_and_capability() -> None:
    mat = np.eye(2, dtype=complex)
    be = BlockEncoding(op=NumpyMatrixOperator(mat), alpha=1.0, resources=ResourceEstimate())
    assert not be.can_export_circuit()
    with pytest.raises(ValueError, match="No circuit recipe"):
        be.build_circuit()

    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    be_no_cap = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=False),
    )
    with pytest.raises(ValueError, match="Circuit export not supported"):
        be_no_cap.build_circuit()


def test_export_openqasm_from_recipe() -> None:
    mat = np.eye(2, dtype=complex)
    circ = Circuit(num_qubits=1)
    circ.add("rx", [0], [0.25])
    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    qasm = be.export_openqasm(flavor="qasm3", optimize=False)
    assert "rx(0.25) q[0];" in qasm


def test_build_circuit_without_optimize_keeps_gates() -> None:
    mat = np.eye(2, dtype=complex)
    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    circ.add("h", [0])
    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    built = be.build_circuit(optimize=False)
    assert len(built.gates) == 2


def test_build_circuit_with_optimize_simplifies() -> None:
    mat = np.eye(2, dtype=complex)
    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    circ.add("h", [0])
    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    built = be.build_circuit()
    assert built.gates == []


def test_package_exports_are_available() -> None:
    assert hasattr(blockflow, "BlockEncoding")
    assert hasattr(blockflow, "Circuit")
