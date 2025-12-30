from __future__ import annotations
from typing import Literal, Optional
from .circuit import Circuit, Gate

QasmFlavor = Literal["qasm2", "qasm3"]

def to_openqasm(circ: Circuit, *, flavor: QasmFlavor = "qasm3") -> str:
    """
    Minimal OpenQASM exporter for a small gate set.
    Extend this map as your recipes and optimizers grow.
    """
    circ.validate()
    needs_classical = any(g.name.lower() == "measure" for g in circ.gates)
    if flavor == "qasm2":
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circ.num_qubits}];"]
        if needs_classical:
            lines.append(f"creg c[{circ.num_qubits}];")
    else:
        lines = ["OPENQASM 3.0;", f"qubit[{circ.num_qubits}] q;"]
        if needs_classical:
            lines.append(f"bit[{circ.num_qubits}] c;")

    for g in circ.gates:
        lines.append(_emit_gate(g, flavor, classical_reg="c" if needs_classical else None))
    return "\n".join(lines) + "\n"

def _emit_gate(g: Gate, flavor: QasmFlavor, classical_reg: Optional[str]) -> str:
    n = g.name.lower()

    if g.controls:
        return _emit_controlled_gate(g, flavor)

    if n in {"h", "x", "y", "z", "s", "t"} and len(g.qubits) == 1 and len(g.params) == 0:
        qs = ", ".join([f"q[{i}]" for i in g.qubits])
        return f"{n} {qs};"

    if n in {"cx", "cz", "swap"} and len(g.qubits) == 2 and len(g.params) == 0:
        qs = ", ".join([f"q[{i}]" for i in g.qubits])
        return f"{n} {qs};"

    if n in {"rx", "ry", "rz"} and len(g.qubits) == 1 and len(g.params) == 1:
        qs = ", ".join([f"q[{i}]" for i in g.qubits])
        return f"{n}({g.params[0]}) {qs};"

    if n == "measure" and len(g.qubits) == 1:
        qs = ", ".join([f"q[{i}]" for i in g.qubits])
        if classical_reg is None:
            raise ValueError("measure export requires a classical register")
        if flavor == "qasm2":
            return f"measure {qs} -> {classical_reg}[{g.qubits[0]}];"
        return f"{classical_reg}[{g.qubits[0]}] = measure {qs};"

    raise ValueError(f"Unsupported gate for QASM export: {g}")


def _emit_controlled_gate(g: Gate, flavor: QasmFlavor) -> str:
    n = g.name.lower()
    controls = g.controls
    if n == "measure":
        raise ValueError("measure cannot be controlled")
    if flavor == "qasm2":
        return _emit_controlled_qasm2(g, n)
    return _emit_controlled_qasm3(g, n, controls)


def _emit_controlled_qasm3(g: Gate, n: str, controls: tuple[int, ...]) -> str:
    gate_call = _format_gate_call(n, g.params)
    ctrl_prefix = "ctrl" if len(controls) == 1 else f"ctrl({len(controls)})"
    qs = ", ".join([f"q[{i}]" for i in controls + g.qubits])
    return f"{ctrl_prefix} @ {gate_call} {qs};"


def _emit_controlled_qasm2(g: Gate, n: str) -> str:
    if len(g.params) != 0:
        raise ValueError("Controlled parameterized gates are not supported in qasm2")
    if len(g.qubits) != 1:
        raise ValueError("Controlled multi-target gates are not supported in qasm2")
    if len(g.controls) == 1 and n in {"x", "y", "z", "h", "s", "t"}:
        gate_name = f"c{n}"
        qs = ", ".join([f"q[{i}]" for i in (g.controls + g.qubits)])
        return f"{gate_name} {qs};"
    if len(g.controls) == 2 and n == "x":
        qs = ", ".join([f"q[{i}]" for i in (g.controls + g.qubits)])
        return f"ccx {qs};"
    raise ValueError("Controlled gate not supported in qasm2")


def _format_gate_call(name: str, params: tuple[float, ...]) -> str:
    if not params:
        return name
    if len(params) == 1:
        return f"{name}({params[0]})"
    raise ValueError("Unsupported parameter count for gate")
