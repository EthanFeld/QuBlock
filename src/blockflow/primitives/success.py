from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SuccessModel:
    """
    success_prob is the probability of landing in the postselection subspace
    for one use of the primitive, if it is postselected.
    If the primitive is unitary with no postselection, set success_prob = 1.0.
    """
    success_prob: float = 1.0
    notes: Optional[str] = None
