from __future__ import annotations

import math
import pytest

from blockflow.compile.circuit import Circuit
from blockflow.compile.optimizers import OptimizationOptions, optimize_circuit
from blockflow.primitives.recipe import StaticCircuitRecipe, WireSpec


def test_circuit_add_and_copy() -> None:
    circ = Circuit(num_qubits=2, name="demo")
    circ.add("h", [0])
    circ.add("cx", [0, 1])
    copied = circ.copy()

    assert copied is not circ
    assert copied.gates is not circ.gates
    assert copied.gates == circ.gates
    assert copied.name == "demo"


def test_circuit_add_validation() -> None:
    circ = Circuit(num_qubits=2)
    with pytest.raises(ValueError, match="at least one qubit"):
        circ.add("h", [])
    with pytest.raises(TypeError, match="name must be a string"):
        circ.add(123, [0])
    with pytest.raises(TypeError, match="qubits must be a list"):
        circ.add("h", (0,))
    with pytest.raises(TypeError, match="qubit indices must be ints"):
        circ.add("x", [0.5])
    with pytest.raises(ValueError, match="out of range"):
        circ.add("x", [2])
    with pytest.raises(ValueError, match="out of range"):
        circ.add("x", [1], controls=[2])
    with pytest.raises(TypeError, match="control indices must be ints"):
        circ.add("x", [1], controls=[0.5])
    with pytest.raises(ValueError, match="unique"):
        circ.add("cx", [0, 0])
    with pytest.raises(ValueError, match="unique"):
        circ.add("x", [0], controls=[0])
    with pytest.raises(TypeError, match="params must be a list"):
        circ.add("rx", [0], params=(0.5,))
    with pytest.raises(TypeError, match="params must be real numbers"):
        circ.add("rx", [0], params=["theta"])
    with pytest.raises(ValueError, match="finite"):
        circ.add("rz", [0], params=[float("inf")])
    with pytest.raises(TypeError, match="controls must be a list"):
        circ.add("x", [1], controls=(0,))


def test_circuit_add_controlled() -> None:
    circ = Circuit(num_qubits=2)
    circ.add_controlled("x", controls=[0], targets=[1])
    gate = circ.gates[0]
    assert gate.controls == (0,)
    assert gate.qubits == (1,)


def test_optimize_self_inverse_cancels() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    circ.add("h", [0])
    optimized = optimize_circuit(circ)
    assert optimized.gates == []


def test_optimize_rotation_merge_to_zero() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("rz", [0], [0.1])
    circ.add("rz", [0], [-0.1])
    optimized = optimize_circuit(circ)
    assert optimized.gates == []


def test_optimize_rotation_merge_nonzero() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("rx", [0], [0.25])
    circ.add("rx", [0], [0.5])
    optimized = optimize_circuit(circ)
    assert len(optimized.gates) == 1
    assert optimized.gates[0].name.lower() == "rx"
    assert abs(optimized.gates[0].params[0] - 0.75) < 1e-12


def test_optimize_rotation_wraps_2pi() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("rz", [0], [math.pi])
    circ.add("rz", [0], [math.pi])
    optimized = optimize_circuit(circ)
    assert optimized.gates == []


def test_optimize_disabled_returns_copy() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("x", [0])
    optimized = optimize_circuit(circ, OptimizationOptions(peephole=False))
    assert optimized is not circ
    assert optimized.gates == circ.gates


def test_static_recipe_applies_optimizer() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    circ.add("h", [0])
    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    built = recipe.build(optimize=True)
    assert built.gates == []


def test_circuit_init_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Circuit(num_qubits=-1)
    with pytest.raises(TypeError, match="non-negative"):
        Circuit(num_qubits=1.5)


def test_circuit_validate_rejects_non_gate() -> None:
    circ = Circuit(num_qubits=1)
    circ.gates.append("not-a-gate")
    with pytest.raises(TypeError, match="Gate"):
        circ.validate()
