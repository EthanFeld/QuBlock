from __future__ import annotations

import pytest

from blockflow.compile.circuit import Circuit, Gate
from blockflow.compile.export_qasm import _emit_gate, to_openqasm


def test_qasm3_measurement_includes_classical_reg() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("h", [0])
    circ.add("cx", [0, 1])
    circ.add("measure", [1])
    qasm = to_openqasm(circ, flavor="qasm3")

    assert "OPENQASM 3.0;" in qasm
    assert "qubit[2] q;" in qasm
    assert "bit[2] c;" in qasm
    assert "c[1] = measure q[1];" in qasm


def test_qasm2_measurement_includes_creg() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("x", [0])
    circ.add("measure", [0])
    qasm = to_openqasm(circ, flavor="qasm2")

    assert "OPENQASM 2.0;" in qasm
    assert 'include "qelib1.inc";' in qasm
    assert "creg c[2];" in qasm
    assert "measure q[0] -> c[0];" in qasm


def test_qasm3_controlled_x() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("x", [1], controls=[0])
    qasm = to_openqasm(circ, flavor="qasm3")
    assert "ctrl @ x q[0], q[1];" in qasm


def test_qasm3_multi_controlled_x() -> None:
    circ = Circuit(num_qubits=3)
    circ.add("x", [2], controls=[0, 1])
    qasm = to_openqasm(circ, flavor="qasm3")
    assert "ctrl(2) @ x q[0], q[1], q[2];" in qasm


def test_qasm3_controlled_param_gate() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("rx", [1], params=[0.25], controls=[0])
    qasm = to_openqasm(circ, flavor="qasm3")
    assert "ctrl @ rx(0.25) q[0], q[1];" in qasm


def test_qasm2_controlled_x_mapping() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("x", [1], controls=[0])
    qasm = to_openqasm(circ, flavor="qasm2")
    assert "cx q[0], q[1];" in qasm


def test_qasm2_multi_controlled_x_raises() -> None:
    circ = Circuit(num_qubits=3)
    circ.add("x", [2], controls=[0, 1])
    qasm = to_openqasm(circ, flavor="qasm2")
    assert "ccx q[0], q[1], q[2];" in qasm


def test_qasm2_controlled_gate_not_supported() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("u", [1], controls=[0])
    with pytest.raises(ValueError, match="not supported"):
        to_openqasm(circ, flavor="qasm2")


def test_emit_gate_requires_classical_for_measure() -> None:
    gate = Gate(name="measure", qubits=(0,), params=())
    with pytest.raises(ValueError, match="classical register"):
        _emit_gate(gate, "qasm3", classical_reg=None)


def test_qasm2_controlled_param_gate_raises() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("rx", [1], params=[0.25], controls=[0])
    with pytest.raises(ValueError, match="parameterized"):
        to_openqasm(circ, flavor="qasm2")


def test_qasm2_controlled_multi_target_raises() -> None:
    circ = Circuit(num_qubits=3)
    circ.add("swap", [1, 2], controls=[0])
    with pytest.raises(ValueError, match="multi-target"):
        to_openqasm(circ, flavor="qasm2")


def test_controlled_measure_raises() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("measure", [0], controls=[1])
    with pytest.raises(ValueError, match="measure cannot be controlled"):
        to_openqasm(circ, flavor="qasm3")


def test_qasm3_controlled_param_count_raises() -> None:
    circ = Circuit(num_qubits=2)
    circ.add("rx", [1], params=[0.1, 0.2], controls=[0])
    with pytest.raises(ValueError, match="parameter count"):
        to_openqasm(circ, flavor="qasm3")


def test_export_unsupported_gate_raises() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("foo", [0])
    with pytest.raises(ValueError, match="Unsupported gate"):
        to_openqasm(circ, flavor="qasm3")
