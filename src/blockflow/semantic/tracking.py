from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RunReport:
    uses: int = 0
    cumulative_success_prob: float = 1.0
    ancilla_clean_peak: int = 0
    ancilla_dirty_peak: int = 0

    def include_use(self, *, success_prob: float, anc_clean: int, anc_dirty: int) -> None:
        self.uses += 1
        self.cumulative_success_prob *= float(success_prob)
        self.ancilla_clean_peak = max(self.ancilla_clean_peak, int(anc_clean))
        self.ancilla_dirty_peak = max(self.ancilla_dirty_peak, int(anc_dirty))
