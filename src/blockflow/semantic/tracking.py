from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from ..primitives.resources import ResourceEstimate


@dataclass
class RunReport:
    uses: int = 0
    cumulative_success_prob: float = 1.0
    ancilla_clean_peak: int = 0
    ancilla_dirty_peak: int = 0
    normalization_product: float = 1.0
    cumulative_error_bound: float = 0.0
    postselections: int = 0
    constraint_checks: int = 0
    unverified_constraints: int = 0
    conditional_error_bound_valid: bool = True
    resources: ResourceEstimate = ResourceEstimate()

    def include_use(
        self,
        *,
        success_prob: float,
        anc_clean: int,
        anc_dirty: int,
        alpha: float = 1.0,
        epsilon: float = 0.0,
        resources: ResourceEstimate | None = None,
        postselected: bool = False,
        constraint_verified: bool | None = None,
    ) -> None:
        if isinstance(success_prob, bool) or not isinstance(success_prob, numbers.Real):
            raise TypeError("success_prob must be a real number between 0 and 1")
        if not math.isfinite(success_prob):
            raise ValueError("success_prob must be finite")
        if success_prob < 0.0 or success_prob > 1.0:
            raise ValueError("success_prob must be between 0 and 1")
        for field_name, value in (("alpha", alpha), ("epsilon", epsilon)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{field_name} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        for field_name, value in (("anc_clean", anc_clean), ("anc_dirty", anc_dirty)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{field_name} must be a non-negative int")
            if value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")
        self.uses += 1
        self.cumulative_success_prob *= float(success_prob)
        self.ancilla_clean_peak = max(self.ancilla_clean_peak, int(anc_clean))
        self.ancilla_dirty_peak = max(self.ancilla_dirty_peak, int(anc_dirty))
        self.normalization_product *= float(alpha)
        self.cumulative_error_bound += float(epsilon)
        self.postselections += int(postselected)
        if postselected and epsilon > 0:
            self.conditional_error_bound_valid = False
        if constraint_verified is not None:
            self.constraint_checks += 1
            self.unverified_constraints += int(not constraint_verified)
        if resources is not None:
            self.resources = self.resources.combine(resources)

    @property
    def constraints_verified(self) -> bool:
        return self.unverified_constraints == 0

    @property
    def expected_trials(self) -> float:
        if self.cumulative_success_prob == 0.0:
            return math.inf
        return 1.0 / self.cumulative_success_prob

    def trials_for_confidence(self, confidence: float) -> int | float:
        if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
            raise TypeError("confidence must be a real number between 0 and 1")
        if not math.isfinite(confidence) or confidence <= 0.0 or confidence >= 1.0:
            raise ValueError("confidence must be strictly between 0 and 1")
        if self.cumulative_success_prob == 0.0:
            return math.inf
        if self.cumulative_success_prob == 1.0:
            return 1
        return math.ceil(
            math.log1p(-float(confidence)) / math.log1p(-self.cumulative_success_prob)
        )
