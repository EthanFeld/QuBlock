from __future__ import annotations
from dataclasses import dataclass
import math
import numbers

@dataclass
class RunReport:
    uses: int = 0
    cumulative_success_prob: float = 1.0
    ancilla_clean_peak: int = 0
    ancilla_dirty_peak: int = 0

    def include_use(self, *, success_prob: float, anc_clean: int, anc_dirty: int) -> None:
        if isinstance(success_prob, bool) or not isinstance(success_prob, numbers.Real):
            raise TypeError("success_prob must be a real number between 0 and 1")
        if not math.isfinite(success_prob):
            raise ValueError("success_prob must be finite")
        if success_prob < 0.0 or success_prob > 1.0:
            raise ValueError("success_prob must be between 0 and 1")
        for field_name, value in (("anc_clean", anc_clean), ("anc_dirty", anc_dirty)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{field_name} must be a non-negative int")
            if value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")
        self.uses += 1
        self.cumulative_success_prob *= float(success_prob)
        self.ancilla_clean_peak = max(self.ancilla_clean_peak, int(anc_clean))
        self.ancilla_dirty_peak = max(self.ancilla_dirty_peak, int(anc_dirty))
