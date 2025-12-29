from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Capabilities:
    supports_adjoint: bool = True
    supports_controlled: bool = False
    supports_power: bool = False
    supports_circuit_recipe: bool = False
