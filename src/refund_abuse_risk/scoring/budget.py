"""Refund-budget pressure — advisory fields + optional effect severity floor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from refund_abuse_risk.schemas.models import RefundEffect

_EFFECT_SEVERITY: dict[RefundEffect, int] = {
    RefundEffect.REFUND_AUTO_GRANT: 0,
    RefundEffect.REFUND_STEP_UP: 1,
    RefundEffect.REFUND_MANUAL_REVIEW: 2,
    RefundEffect.REFUND_BLOCK: 3,
}


@dataclass
class RefundBudgetHint:
    enabled: bool
    pressure: str  # ok | elevated | exhausted
    suggested_effect: str | None
    refund_count_30d: float
    refund_gmv_pct_30d: float
    max_refund_count_30d: float
    max_refund_gmv_pct_30d: float
    max_auto_grants_30d: float
    floored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pressure": self.pressure,
            "suggested_effect": self.suggested_effect,
            "refund_count_30d": self.refund_count_30d,
            "refund_gmv_pct_30d": self.refund_gmv_pct_30d,
            "max_refund_count_30d": self.max_refund_count_30d,
            "max_refund_gmv_pct_30d": self.max_refund_gmv_pct_30d,
            "max_auto_grants_30d": self.max_auto_grants_30d,
            "floored": self.floored,
        }


def effect_severity(effect: RefundEffect | str | None) -> int:
    if effect is None:
        return -1
    try:
        e = effect if isinstance(effect, RefundEffect) else RefundEffect(str(effect))
    except ValueError:
        return -1
    return int(_EFFECT_SEVERITY.get(e, -1))


def evaluate_refund_budget(
    features: dict[str, Any],
    cfg: dict[str, Any] | None,
) -> RefundBudgetHint:
    c = cfg or {}
    if not c.get("enabled", True):
        return RefundBudgetHint(
            enabled=False,
            pressure="ok",
            suggested_effect=None,
            refund_count_30d=0.0,
            refund_gmv_pct_30d=0.0,
            max_refund_count_30d=0.0,
            max_refund_gmv_pct_30d=0.0,
            max_auto_grants_30d=0.0,
        )
    count = float(features.get("user_refund_count_30d", 0) or 0)
    gmv_pct = float(features.get("user_refund_gmv_pct_30d", 0) or 0)
    max_count = float(c.get("max_refund_count_30d", 5))
    max_gmv = float(c.get("max_refund_gmv_pct_30d", 0.35))
    max_auto = float(c.get("max_auto_grants_30d", 3))
    pressure = "ok"
    if count >= max_count or gmv_pct >= max_gmv:
        pressure = "exhausted"
    elif count >= max_auto or gmv_pct >= (max_gmv * 0.7):
        pressure = "elevated"
    effects = c.get("pressure_effects") or {}
    suggested = effects.get(pressure)
    return RefundBudgetHint(
        enabled=True,
        pressure=pressure,
        suggested_effect=str(suggested) if suggested else None,
        refund_count_30d=count,
        refund_gmv_pct_30d=gmv_pct,
        max_refund_count_30d=max_count,
        max_refund_gmv_pct_30d=max_gmv,
        max_auto_grants_30d=max_auto,
    )


def apply_budget_effect_floor(
    current: RefundEffect,
    budget: RefundBudgetHint,
    cfg: dict[str, Any] | None,
) -> tuple[RefundEffect, RefundBudgetHint, list[str]]:
    """
    Raise final effect to budget suggestion when stricter.

    Returns (effect, updated_budget_hint, reason_codes).
    """
    c = cfg or {}
    if not budget.enabled or not c.get("apply_as_effect_floor", True):
        return current, budget, []
    if not budget.suggested_effect:
        return current, budget, []
    try:
        floor = RefundEffect(str(budget.suggested_effect))
    except ValueError:
        return current, budget, []
    if effect_severity(floor) <= effect_severity(current):
        return current, budget, []
    budget.floored = True
    return floor, budget, ["BUDGET_FLOOR"]
