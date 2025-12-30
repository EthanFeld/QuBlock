from __future__ import annotations
from dataclasses import dataclass
import math
import numbers
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

    def __post_init__(self) -> None:
        if isinstance(self.success_prob, bool) or not isinstance(self.success_prob, numbers.Real):
            raise TypeError("success_prob must be a real number between 0 and 1")
        if not math.isfinite(self.success_prob):
            raise ValueError("success_prob must be finite")
        if self.success_prob < 0.0 or self.success_prob > 1.0:
            raise ValueError("success_prob must be between 0 and 1")
