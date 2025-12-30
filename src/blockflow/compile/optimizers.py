from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
from .circuit import Circuit, Gate

@dataclass
class OptimizationOptions:
    peephole: bool = True

_SELF_INVERSE = {"h", "x", "y", "z", "cx", "cz", "swap"}
_ROTATIONS = {"rx", "ry", "rz"}

def optimize_circuit(circ: Circuit, opts: Optional[OptimizationOptions] = None) -> Circuit:
    if opts is None:
        opts = OptimizationOptions()
    if not opts.peephole:
        return circ.copy()

    optimized = []
    for g in circ.gates:
        name = g.name.lower()
        if optimized:
            prev = optimized[-1]
            prev_name = prev.name.lower()
            if (
                name in _SELF_INVERSE
                and name == prev_name
                and g.qubits == prev.qubits
                and g.controls == prev.controls
                and not g.params
                and not prev.params
            ):
                optimized.pop()
                continue
            if (
                name in _ROTATIONS
                and name == prev_name
                and g.qubits == prev.qubits
                and g.controls == prev.controls
                and len(g.params) == 1
                and len(prev.params) == 1
            ):
                angle = prev.params[0] + g.params[0]
                angle = math.remainder(angle, 2 * math.pi)
                if math.isclose(angle, 0.0, abs_tol=1e-12):
                    optimized.pop()
                else:
                    optimized[-1] = Gate(
                        name=prev.name,
                        qubits=prev.qubits,
                        params=(angle,),
                        controls=prev.controls,
                    )
                continue
        optimized.append(g)

    return Circuit(num_qubits=circ.num_qubits, gates=optimized, name=circ.name)
