from __future__ import annotations

import numpy as np
import pytest

from blockflow import Circuit, StaticStatePreparationRecipe, VectorEncoding, WireSpec
from blockflow.primitives.capabilities import Capabilities
from blockflow.primitives.resources import ResourceEstimate


def test_vector_encoding_validation_errors() -> None:
    with pytest.raises(TypeError, match="alpha"):
        VectorEncoding(vec=np.array([1.0]), alpha="bad", resources=ResourceEstimate())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        VectorEncoding(vec=np.array([1.0]), alpha=0.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="epsilon"):
        VectorEncoding(vec=np.array([1.0]), alpha=1.0, resources=ResourceEstimate(), epsilon=-0.1)
    with pytest.raises(ValueError, match="1D"):
        VectorEncoding(vec=np.eye(2), alpha=1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="non-empty"):
        VectorEncoding(vec=np.array([]), alpha=1.0, resources=ResourceEstimate())


def test_vector_encoding_from_vector_errors() -> None:
    with pytest.raises(ValueError, match="1D"):
        VectorEncoding.from_vector(np.eye(2), resources=ResourceEstimate())
    with pytest.raises(ValueError, match="nonzero"):
        VectorEncoding.from_vector(np.zeros(2), resources=ResourceEstimate())


def test_vector_encoding_semantic_state_and_dimension() -> None:
    ve = VectorEncoding(vec=np.array([3.0, 4.0]), alpha=5.0, resources=ResourceEstimate())
    assert ve.dimension == 2
    assert np.allclose(ve.semantic_state(), np.array([0.6, 0.8]))


def test_vector_encoding_recipe_build_and_checks() -> None:
    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    recipe = StaticStatePreparationRecipe(WireSpec(system_qubits=1), circ)
    ve = VectorEncoding(
        vec=np.array([1.0, 0.0]),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    built = ve.build_circuit(optimize=True)
    assert built.num_qubits == 1

    ve_no_cap = VectorEncoding(
        vec=np.array([1.0, 0.0]),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=False),
    )
    with pytest.raises(ValueError, match="Circuit export not supported"):
        ve_no_cap.build_circuit()

    ve_no_recipe = VectorEncoding(vec=np.array([1.0, 0.0]), alpha=1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="No state preparation recipe"):
        ve_no_recipe.build_circuit()

    circ_bad = Circuit(num_qubits=2)
    recipe_bad = StaticStatePreparationRecipe(WireSpec(system_qubits=1), circ_bad)
    ve_bad = VectorEncoding(
        vec=np.array([1.0, 0.0]),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe_bad,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    with pytest.raises(ValueError, match="WireSpec"):
        ve_bad.build_circuit(optimize=False)

    recipe_anc = StaticStatePreparationRecipe(WireSpec(system_qubits=1, ancilla_clean=1), circ_bad)
    ve_anc = VectorEncoding(
        vec=np.array([1.0, 0.0]),
        alpha=1.0,
        resources=ResourceEstimate(ancilla_qubits_clean=0),
        recipe=recipe_anc,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    with pytest.raises(ValueError, match="clean ancillas"):
        ve_anc.build_circuit(optimize=False)

    recipe_dirty = StaticStatePreparationRecipe(WireSpec(system_qubits=1, ancilla_dirty=1), circ_bad)
    ve_dirty = VectorEncoding(
        vec=np.array([1.0, 0.0]),
        alpha=1.0,
        resources=ResourceEstimate(ancilla_qubits_dirty=0),
        recipe=recipe_dirty,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    with pytest.raises(ValueError, match="dirty ancillas"):
        ve_dirty.build_circuit(optimize=False)
