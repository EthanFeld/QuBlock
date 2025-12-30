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


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _unitary_from_circuit_1q(circ: Circuit) -> np.ndarray:
    unitary = np.eye(2, dtype=complex)
    for gate in circ.gates:
        name = gate.name.lower()
        if name == "rz":
            gate_u = _rz(gate.params[0])
        elif name == "ry":
            gate_u = _ry(gate.params[0])
        else:
            raise ValueError(f"Unsupported gate for unitary check: {gate.name}")
        unitary = gate_u @ unitary
    return unitary


def _assert_global_phase_equiv(lhs: np.ndarray, rhs: np.ndarray, *, atol: float = 1e-7) -> None:
    delta = lhs @ rhs.conj().T
    phase = np.angle(np.trace(delta))
    assert np.allclose(delta, np.exp(1j * phase) * np.eye(2), atol=atol)


def _assert_global_phase_equiv_matrix(lhs: np.ndarray, rhs: np.ndarray, *, atol: float = 1e-7) -> None:
    denom = np.vdot(rhs, rhs)
    if abs(denom) < 1e-12:
        assert np.allclose(lhs, rhs, atol=atol)
        return
    phase = np.vdot(rhs, lhs) / denom
    if abs(phase) > 0:
        phase = phase / abs(phase)
    assert np.allclose(lhs, phase * rhs, atol=atol)


def _gate_matrix(name: str, params: tuple[float, ...]) -> np.ndarray:
    if name == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if name == "y":
        return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    if name == "z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    if name == "ry":
        return _ry(params[0])
    if name == "rz":
        return _rz(params[0])
    raise ValueError(f"Unsupported gate for unitary check: {name}")


def _pauli_matrix(labels: str) -> np.ndarray:
    lookup = {
        "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    mat = lookup[labels[0]]
    for label in labels[1:]:
        mat = np.kron(lookup[label], mat)
    return mat


def _apply_gate_to_state(
    state: np.ndarray,
    *,
    gate: np.ndarray,
    target: int,
    controls: tuple[int, ...],
) -> np.ndarray:
    n = int(np.log2(state.shape[0]))
    new_state = state.copy()
    for i in range(2**n):
        if (i >> target) & 1:
            continue
        if any(((i >> c) & 1) == 0 for c in controls):
            continue
        j = i | (1 << target)
        a0 = state[i]
        a1 = state[j]
        new_state[i] = gate[0, 0] * a0 + gate[0, 1] * a1
        new_state[j] = gate[1, 0] * a0 + gate[1, 1] * a1
    return new_state


def _unitary_from_circuit(circ: Circuit) -> np.ndarray:
    dim = 2 ** circ.num_qubits
    unitary = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        state = np.zeros(dim, dtype=complex)
        state[col] = 1.0
        for gate in circ.gates:
            gate_u = _gate_matrix(gate.name.lower(), gate.params)
            state = _apply_gate_to_state(
                state,
                gate=gate_u,
                target=gate.qubits[0],
                controls=gate.controls,
            )
        unitary[:, col] = state
    return unitary


def _extract_block(unitary: np.ndarray, *, ancillas: tuple[int, ...], num_qubits: int) -> np.ndarray:
    keep = [
        idx
        for idx in range(2**num_qubits)
        if all(((idx >> anc) & 1) == 0 for anc in ancillas)
    ]
    return unitary[np.ix_(keep, keep)]


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


def test_can_export_with_recipe() -> None:
    mat = np.eye(2, dtype=complex)
    circ = Circuit(num_qubits=1)
    circ.add("h", [0])
    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    assert be.can_export_circuit()


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


def test_build_circuit_synthesizes_from_matrix() -> None:
    phi = 0.2
    theta = 1.1
    lam = -0.4
    unitary = _rz(phi) @ _ry(theta) @ _rz(lam)
    alpha = 1.7
    mat = alpha * unitary

    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=alpha,
        resources=ResourceEstimate(),
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    assert be.can_export_circuit()
    circ = be.build_circuit(optimize=False)
    recovered = _unitary_from_circuit_1q(circ)
    _assert_global_phase_equiv(recovered, unitary)


def test_build_circuit_synthesizes_block_encoding_from_matrix() -> None:
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
    circ = be.build_circuit(optimize=False)
    unitary = _unitary_from_circuit(circ)
    ancillas = tuple(range(1, circ.num_qubits))
    block = _extract_block(unitary, ancillas=ancillas, num_qubits=circ.num_qubits)
    _assert_global_phase_equiv_matrix(block, mat / alpha)


def test_build_circuit_wraps_matrix_synthesis_error() -> None:
    mat = np.eye(3, dtype=complex)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    with pytest.raises(ValueError, match="matrix synthesis failed"):
        be.build_circuit()


@pytest.mark.parametrize("strategy", ["prep_select", "sparse"])
def test_build_circuit_synthesizes_block_encoding_n_qubits(strategy: str) -> None:
    coeffs = [
        0.3 + 0.1j,
        -0.2j,
        0.15 - 0.05j,
    ]
    labels = ["XI", "YZ", "IZ"]
    mat = sum(c * _pauli_matrix(lbl) for c, lbl in zip(coeffs, labels, strict=True))
    alpha = float(np.sum(np.abs(coeffs)))

    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=alpha,
        resources=ResourceEstimate(ancilla_qubits_clean=8),
        capabilities=Capabilities(supports_circuit_recipe=True),
        synthesis_strategy=strategy,
    )
    circ = be.build_circuit(optimize=False)
    unitary = _unitary_from_circuit(circ)
    ancillas = tuple(range(2, circ.num_qubits))
    block = _extract_block(unitary, ancillas=ancillas, num_qubits=circ.num_qubits)
    _assert_global_phase_equiv_matrix(block, mat / alpha)


def test_block_encoding_strategy_respects_resources() -> None:
    coeffs = np.array([0.25, 0.25, 0.5], dtype=complex)
    labels = ["XI", "IZ", "ZX"]
    mat = sum(c * _pauli_matrix(label) for c, label in zip(coeffs, labels, strict=True))
    alpha = float(np.sum(np.abs(coeffs)))

    be_prep = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=alpha,
        resources=ResourceEstimate(ancilla_qubits_clean=2),
        capabilities=Capabilities(supports_circuit_recipe=True),
        synthesis_strategy="prep_select",
    )
    assert be_prep.can_export_circuit()
    circ = be_prep.build_circuit(optimize=False)
    assert circ.num_qubits == 4

    be_sparse = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=alpha,
        resources=ResourceEstimate(ancilla_qubits_clean=2),
        capabilities=Capabilities(supports_circuit_recipe=True),
        synthesis_strategy="sparse",
    )
    assert not be_sparse.can_export_circuit()
    with pytest.raises(ValueError, match="clean ancillas"):
        be_sparse.build_circuit(optimize=False)


def test_block_encoding_validation_errors() -> None:
    mat = np.eye(2, dtype=complex)
    with pytest.raises(ValueError, match="alpha"):
        BlockEncoding(op=NumpyMatrixOperator(mat), alpha=0.0, resources=ResourceEstimate())
    with pytest.raises(TypeError, match="alpha"):
        BlockEncoding(op=NumpyMatrixOperator(mat), alpha="bad", resources=ResourceEstimate())
    with pytest.raises(ValueError, match="finite"):
        BlockEncoding(op=NumpyMatrixOperator(mat), alpha=float("inf"), resources=ResourceEstimate())
    with pytest.raises(ValueError, match="epsilon"):
        BlockEncoding(op=NumpyMatrixOperator(mat), alpha=1.0, resources=ResourceEstimate(), epsilon=-1.0)

    non_square = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="square"):
        BlockEncoding(op=NumpyMatrixOperator(non_square), alpha=1.0, resources=ResourceEstimate())

    with pytest.raises(ValueError, match="synthesis_strategy"):
        BlockEncoding(
            op=NumpyMatrixOperator(mat),
            alpha=1.0,
            resources=ResourceEstimate(),
            synthesis_strategy="bad",  # type: ignore[arg-type]
        )

    class BadShapeOperator:
        @property
        def shape(self):
            return [2, 2]

        @property
        def dtype(self):
            return np.dtype(complex)

        def apply(self, vec: np.ndarray) -> np.ndarray:
            return vec

        def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
            return vec

        def norm_bound(self) -> float:
            return 1.0

    with pytest.raises(TypeError, match="shape"):
        BlockEncoding(op=BadShapeOperator(), alpha=1.0, resources=ResourceEstimate())

    class NonIntShapeOperator:
        @property
        def shape(self) -> tuple[float, int]:
            return (2.5, 2)

        @property
        def dtype(self):
            return np.dtype(complex)

        def apply(self, vec: np.ndarray) -> np.ndarray:
            return vec

        def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
            return vec

        def norm_bound(self) -> float:
            return 1.0

    with pytest.raises(TypeError, match="op.shape"):
        BlockEncoding(op=NonIntShapeOperator(), alpha=1.0, resources=ResourceEstimate())

    class ZeroDimOperator:
        @property
        def shape(self) -> tuple[int, int]:
            return (0, 2)

        @property
        def dtype(self):
            return np.dtype(complex)

        def apply(self, vec: np.ndarray) -> np.ndarray:
            return vec

        def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
            return vec

        def norm_bound(self) -> float:
            return 1.0

    with pytest.raises(ValueError, match="positive"):
        BlockEncoding(op=ZeroDimOperator(), alpha=1.0, resources=ResourceEstimate())


def test_semantic_apply_dimension_checks() -> None:
    mat = np.eye(2, dtype=complex)
    be = BlockEncoding(op=NumpyMatrixOperator(mat), alpha=1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="expected 2"):
        be.semantic_apply(np.array([1.0, 0.0, 0.0], dtype=complex))
    with pytest.raises(ValueError, match="1D"):
        be.semantic_apply(np.eye(2, dtype=complex))

    class BadOperator:
        @property
        def shape(self) -> tuple[int, int]:
            return (2, 2)

        @property
        def dtype(self):
            return np.dtype(complex)

        def apply(self, vec: np.ndarray) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0], dtype=complex)

        def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0], dtype=complex)

        def norm_bound(self) -> float:
            return 1.0

    bad_be = BlockEncoding(op=BadOperator(), alpha=1.0, resources=ResourceEstimate())
    with pytest.raises(ValueError, match="output"):
        bad_be.semantic_apply(np.array([1.0, 0.0], dtype=complex))


def test_semantic_apply_adjoint_executes() -> None:
    mat = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    be = BlockEncoding(op=NumpyMatrixOperator(mat), alpha=1.0, resources=ResourceEstimate())
    out = be.semantic_apply_adjoint(np.array([1.0, 0.0], dtype=complex))
    assert np.allclose(out, np.array([0.0, 1.0], dtype=complex))


def test_build_circuit_validates_wire_spec_and_resources() -> None:
    mat = np.eye(2, dtype=complex)
    circ = Circuit(num_qubits=2)
    circ.add("h", [0])
    recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)
    be = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        recipe=recipe,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    with pytest.raises(ValueError, match="WireSpec"):
        be.build_circuit(optimize=False)

    circ_ok = Circuit(num_qubits=2)
    circ_ok.add("h", [0])
    recipe_with_anc = StaticCircuitRecipe(WireSpec(system_qubits=1, ancilla_clean=1), circ_ok)
    be_insufficient = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(ancilla_qubits_clean=0),
        recipe=recipe_with_anc,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    with pytest.raises(ValueError, match="clean ancillas"):
        be_insufficient.build_circuit(optimize=False)

    recipe_with_dirty = StaticCircuitRecipe(WireSpec(system_qubits=1, ancilla_dirty=1), circ_ok)
    be_dirty = BlockEncoding(
        op=NumpyMatrixOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(ancilla_qubits_dirty=0),
        recipe=recipe_with_dirty,
        capabilities=Capabilities(supports_circuit_recipe=True),
    )
    with pytest.raises(ValueError, match="dirty ancillas"):
        be_dirty.build_circuit(optimize=False)


def test_build_circuit_with_opts_runs_optimizer() -> None:
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
    built = be.build_circuit(optimize=True, opts=blockflow.OptimizationOptions())
    assert built.gates == []


def test_package_exports_are_available() -> None:
    assert hasattr(blockflow, "BlockEncoding")
    assert hasattr(blockflow, "Circuit")


def test_program_append_and_renormalize() -> None:
    mat = np.eye(2, dtype=complex) * 2.0
    be = BlockEncoding(op=NumpyMatrixOperator(mat), alpha=1.0, resources=ResourceEstimate())
    program = Program()
    program.append(ApplyBlockEncodingStep(be))
    state = StateVector(np.array([1.0, 0.0], dtype=complex))
    final_state, _ = SemanticExecutor().run(program, state, renormalize_each_step=True)
    assert np.allclose(final_state.data, np.array([1.0, 0.0], dtype=complex))
