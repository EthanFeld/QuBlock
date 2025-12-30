from __future__ import annotations

import numpy as np
import pytest

from blockflow.primitives.capabilities import Capabilities
from blockflow.primitives.recipe import WireSpec
from blockflow.primitives.resources import ResourceEstimate
from blockflow.primitives.vector_encoding import StaticStatePreparationRecipe, VectorEncoding
from blockflow.compile.circuit import Circuit


def test_vector_encoding_from_vector() -> None:
    vec = np.array([3.0, 4.0])
    ve = VectorEncoding.from_vector(vec, resources=ResourceEstimate())
    assert np.isclose(ve.alpha, 5.0)
    assert np.allclose(ve.semantic_state(), np.array([0.6, 0.8]))


def test_vector_encoding_builds_from_recipe() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    recipe = StaticStatePreparationRecipe(WireSpec(system_qubits=1), circ)
    ve = VectorEncoding(
        vec=np.array([1.0, 0.0]),
        alpha=1.0,
        resources=ResourceEstimate(),
        capabilities=Capabilities(supports_circuit_recipe=True),
        recipe=recipe,
    )
    built = ve.build_circuit()
    assert built.num_qubits == 1
    assert len(built.gates) == 1
    assert ve.can_export_circuit()

    qasm = ve.export_openqasm()
    assert "OPENQASM" in qasm


def test_vector_encoding_validation_errors() -> None:
    with pytest.raises(ValueError, match="1D"):
        VectorEncoding(vec=np.eye(2), alpha=1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="non-empty"):
        VectorEncoding(vec=np.array([]), alpha=1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="must be > 0"):
        VectorEncoding(vec=np.array([1.0]), alpha=0.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="must be >= 0"):
        VectorEncoding(vec=np.array([1.0]), alpha=1.0, epsilon=-1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="nonzero"):
        VectorEncoding.from_vector(np.array([0.0, 0.0]), resources=ResourceEstimate())


def test_vector_encoding_build_circuit_errors() -> None:
    ve = VectorEncoding(vec=np.array([1.0, 0.0]), alpha=1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="No state preparation recipe"):
        ve.build_circuit()

    circ = Circuit(num_qubits=1)
    recipe = StaticStatePreparationRecipe(WireSpec(system_qubits=1), circ)
    ve_no_caps = VectorEncoding(
        vec=np.array([1.0, 0.0]),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
    )
    with pytest.raises(ValueError, match="Circuit export not supported"):
        ve_no_caps.build_circuit()
