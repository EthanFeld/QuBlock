# QuBlock
QuBlock is a small library for representing block encodings, simulating their action
on system statevectors, and exporting recipe-based circuits to OpenQASM.

## Quickstart
```python
import numpy as np

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

# Define a linear operator.
mat = np.array([[0, 1], [1, 0]], dtype=complex)
op = NumpyMatrixOperator(mat)

# Wrap it in a block encoding and run a semantic program.
be = BlockEncoding(op=op, alpha=1.0, resources=ResourceEstimate())
program = Program([ApplyBlockEncodingStep(be)])
state = StateVector(np.array([1.0, 0.0], dtype=complex))
final_state, report = SemanticExecutor().run(program, state)

# Attach a simple circuit recipe and export to QASM.
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
qasm = be_with_recipe.export_openqasm()
```
