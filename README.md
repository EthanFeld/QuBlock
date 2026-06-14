![PyPI - Downloads](https://img.shields.io/pypi/dm/qublock)
# QuBlock
QuBlock is a lightweight Python library for block-encoding primitives, semantic
simulation on statevectors, and OpenQASM export. The pip package name is
`qublock`, and the import namespace is `blockflow`.

## Highlights
- Explicit block-encoding invariants with normalization, error tolerance, and success tracking.
- Fast semantic execution for algorithm validation without circuits.
- Recipe-based circuits with declared wire requirements.
- OpenQASM 2 or 3 export with a minimal gate set.
- Small dependency footprint (NumPy only at runtime).

## Requirements
- Python 3.9+
- NumPy >= 1.22

## Installation
```bash
pip install qublock
```

Optional extras:
```bash
pip install qublock[gpu]    # CuPy backend
pip install qublock[torch]  # PyTorch backend
pip install qublock[sparse] # SciPy sparse operators
pip install -e .[tutorial]  # Qiskit/Jupyter VQLS tutorial
```

## Quickstart
### Semantic execution
```python
import numpy as np

from blockflow import (
    ApplyBlockEncodingStep,
    BlockEncoding,
    NumpyMatrixOperator,
    Program,
    ResourceEstimate,
    SemanticExecutor,
    StateVector,
)

mat = np.array([[0, 1], [1, 0]], dtype=complex)
op = NumpyMatrixOperator(mat)
be = BlockEncoding(op=op, alpha=1.0, resources=ResourceEstimate())

program = Program([ApplyBlockEncodingStep(be)])
state = StateVector(np.array([1.0, 0.0], dtype=complex))
final_state, report = SemanticExecutor().run(program, state, renormalize_each_step=True)
```

### Attach a circuit recipe and export QASM
```python
from blockflow import (
    Capabilities,
    Circuit,
    StaticCircuitRecipe,
    WireSpec,
)

circ = Circuit(num_qubits=1)
circ.add("h", [0])
recipe = StaticCircuitRecipe(WireSpec(system_qubits=1), circ)

be_with_recipe = BlockEncoding(
    op=op,
    alpha=1.0,
    resources=ResourceEstimate(),
    recipe=recipe,
    capabilities=Capabilities(supports_circuit_recipe=True),
)
qasm = be_with_recipe.export_openqasm(flavor="qasm3")
```

## Core concepts
- Linear operators: implement the `LinearOperator` protocol (`shape`, `apply`, `apply_adjoint`).
- Block encodings: `BlockEncoding` wraps an operator with `alpha`, `epsilon`, resources, and recipes.
- Vector encodings: `VectorEncoding` represents normalized state-prep with optional recipes.
- Capabilities and success: `Capabilities` and `SuccessModel` capture supported operations and
  postselection success rates.
- Resources: `ResourceEstimate` tracks ancillas, depth, and gate counts.
- Recipes and circuits: `CircuitRecipe` produces backend-agnostic `Circuit` objects with a `WireSpec`.
- QASM export: `to_openqasm` emits a minimal gate set to QASM2/QASM3.

## Semantic execution
`SemanticExecutor` applies each program step directly to the system statevector.
This is useful for validating algorithmic structure before committing to a
specific circuit implementation.

```python
program = Program([ApplyBlockEncodingStep(be)])
state = StateVector(np.array([1.0, 0.0], dtype=complex))
final_state, report = SemanticExecutor().run(program, state)
```

The `RunReport` accumulates uses, success probabilities, and ancilla peaks.

### Quantum-constrained semantic execution
`ApplyBlockEncodingStep` applies the raw classical map `A` for ordinary linear
algebra experiments. `ApplyBlockEncodingQuantumStep` instead applies the
physically projected block `A / alpha`, while still storing only the system
vector.

```python
from blockflow import ApplyBlockEncodingQuantumStep

program = Program([
    ApplyBlockEncodingQuantumStep(
        be,
        repeat=3,
        adjoint=False,
        postselect=True,
        require_verified=True,
    )
])
final_state, report = SemanticExecutor().run(program, state)
```

For each use, the quantum step:
- verifies `alpha >= ||A||` exactly for built-in dense, diagonal, and
  permutation operators, or from a sufficiently tight declared `norm_bound()`;
- applies `A / alpha` or `A† / alpha` classically;
- computes state-dependent projected-block success probability;
- optionally normalizes the successful postselected branch;
- accumulates normalization, projected-block error, declared quantum resources,
  ancilla peaks, and expected retry cost.

`report.expected_trials` estimates independent runs needed per successful
result. `report.trials_for_confidence(p)` estimates runs needed to see at least
one success with confidence `p`. For approximate encodings, postselection can
amplify state error, so `report.conditional_error_bound_valid` becomes false;
`cumulative_error_bound` then bounds only unnormalized projected-map
composition.

### Scope and scaling
Semantic execution stores only the designated system statevector. Declared
block-encoding ancillas are tracked as metadata and are never added to the
simulated state, so a primitive with `n` system qubits and `a` ancillas uses
`O(2^n)` state memory instead of `O(2^(n+a))`.

The operator implementation still determines application cost. Structured,
sparse, GPU-backed, and matrix-free operators can preserve the ancilla-saving
advantage; an explicit dense `2^n x 2^n` matrix still requires `O(4^n)` storage.

Current quantum semantic programs model one selected projected/postselected
system branch through sequential block uses and adjoints. Algorithms whose
result depends on coherent interference between success and failure ancilla
branches, controlled block encodings, amplitude amplification, or QSVT phase
sequences cannot be reproduced from one system vector and remain unsupported.

### Backend selection and speed flags
Semantic execution defaults to NumPy. You can switch backends without changing
code by setting environment variables:
- `BLOCKFLOW_BACKEND=numpy|cupy|torch|auto` (default: numpy)
- `BLOCKFLOW_DEVICE=cuda` (torch only, optional)
- `BLOCKFLOW_DTYPE=complex64|complex128|float32|float64` (optional)
- `BLOCKFLOW_COPY_STATE=0` to avoid copying the initial state

Example:
```bash
BLOCKFLOW_BACKEND=cupy python your_script.py
```

To apply a block encoding multiple times in one step, use `repeat`:
```python
program = Program([ApplyBlockEncodingStep(be, repeat=5)])
```

## Circuits, recipes, and OpenQASM
Recipes declare required wires and return a backend-agnostic circuit. The block
encoding verifies that the recipe matches its resource claims.

```python
from blockflow import Capabilities, Circuit, StaticCircuitRecipe, WireSpec

circ = Circuit(num_qubits=2)
circ.add("h", [0])
circ.add("cx", [0, 1])

recipe = StaticCircuitRecipe(WireSpec(system_qubits=2), circ)
be = BlockEncoding(
    op=op,
    alpha=1.0,
    resources=ResourceEstimate(),
    recipe=recipe,
    capabilities=Capabilities(supports_circuit_recipe=True),
)

qasm3 = be.export_openqasm(flavor="qasm3")
qasm2 = be.export_openqasm(flavor="qasm2", optimize=False)
```

Export currently supports a minimal gate set (`h`, `x`, `y`, `z`, `s`, `t`,
`cx`, `cz`, `swap`, `rx`, `ry`, `rz`, and `measure`) plus controlled variants in QASM3.

## Matrix to block-encoding synthesis (n-qubit)
If you have a `2^n x 2^n` matrix, QuBlock can synthesize a block-encoding circuit
without attaching a recipe. There are two paths:
- If `A / alpha` is unitary (2x2 only), it synthesizes a 1-qubit circuit that implements it.
- Otherwise, it builds an LCU block encoding using the Pauli expansion of `A`
  (requires `alpha == sum(|Pauli coeffs|)`).

For LCU synthesis you can choose a strategy:
- `prep_select` uses `ceil(log2(m))` ancillas (plus one phase ancilla if needed) and multi-controlled
  select gates, where `m` is the number of nonzero Pauli terms.
- `sparse` uses one ancilla per term and single-controlled select gates.

Both LCU strategies export only to QASM3 because they use controlled rotations.
Pauli decomposition enumerates `4^n` Pauli strings, so automatic matrix
synthesis is intended for small validation/export cases rather than large-scale
semantic simulation.

```python
mat = np.array([[0, 1], [1, 0]], dtype=complex)
be = BlockEncoding(
    op=NumpyMatrixOperator(mat),
    alpha=1.0,
    resources=ResourceEstimate(),
    capabilities=Capabilities(supports_circuit_recipe=True),
    synthesis_strategy="prep_select",  # or "sparse"
)
qasm = be.export_openqasm()
```

## Optimization
Use `optimize_circuit` to apply simple peephole optimizations, or disable it
when you need a 1:1 recipe export.

```python
optimized = be.build_circuit(optimize=True)
raw = be.build_circuit(optimize=False)
```

## Examples and notebooks
- `notebooks/lcu_demo.ipynb` walks through LCU synthesis and QASM export.
- `notebooks/vqls_qiskit_tutorial.ipynb` prototypes VQLS with a Qiskit ansatz,
  classical optimization, and QuBlock quantum-constraint tracking.

## Development
Tests run with coverage enforcement:

```bash
pytest
```

Ruff is configured for linting and import sorting:

```bash
ruff check .
```

## Optional Qiskit cross-check
`tests/test_qiskit_integration.py` compares QASM output against Qiskit statevector
simulation. Install Qiskit to enable those tests.
