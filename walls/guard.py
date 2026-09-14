from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .detector import WallSnapshot, WallLevel, WallSide


@dataclass
class WallGuardParams:
    min_density: float = 1.4
    max_distance_frac: float = 0.008
    max_negative_decay: float = 0.30
    min_touch_count: int = 2
    max_negative_health_change: float = 0.25
    max_negative_health_velocity: float = 0.20
    allowed_health_states: tuple[str, ...] = ("stable", "building")


@dataclass
class GuardDecision:
    original_signal: str
    signal: str
    triggered: bool
    reason: Optional[str] = None
    side: Optional[str] = None
    wall: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "original_signal": self.original_signal,
            "signal": self.signal,
            "triggered": self.triggered,
            "reason": self.reason,
            "side": self.side,
            "wall": self.wall,
        }


class WallGuard:
    """Evaluates wall snapshots to optionally suppress execution signals."""

    def __init__(self, params: Optional[WallGuardParams] = None) -> None:
        self.params = params or WallGuardParams()

    def evaluate(self, signal: str, snapshot: Optional[WallSnapshot]) -> GuardDecision:
        original = (signal or "flat").lower()
        if original not in {"long", "short"}:
            return GuardDecision(original_signal=original, signal=original, triggered=False)
        if snapshot is None or not snapshot.ready:
            return GuardDecision(original_signal=original, signal=original, triggered=False)

        target_wall: Optional[WallLevel]
        side: Optional[str]
        if original == "short":
            target_wall = snapshot.put_wall
            side = "put"
        else:
            target_wall = snapshot.call_wall
            side = "call"

        if target_wall is None:
            return GuardDecision(original_signal=original, signal=original, triggered=False)

        wall_dict = target_wall.to_dict()
        summary = (
            f"{side} wall @{target_wall.strike:.0f} density={target_wall.density:.2f} "
            f"decay={target_wall.decay_rate:.2f} touches={target_wall.touch_count} "
            f"health={target_wall.health_state} Δ={target_wall.health_change:+.2f} "
            f"vel={target_wall.health_velocity:+.2f}"
        )
        should_block = self._should_block(target_wall)
        if should_block:
            return GuardDecision(
                original_signal=original,
                signal="flat",
                triggered=True,
                reason=summary,
                side=side,
                wall=wall_dict,
            )

        return GuardDecision(
            original_signal=original,
            signal=original,
            triggered=False,
            reason=summary,
            side=side,
            wall=wall_dict,
        )

    def _should_block(self, wall: WallLevel) -> bool:
        params = self.params
        if not wall.active:
            return False
        if wall.density < params.min_density:
            return False
        if wall.distance_frac > params.max_distance_frac:
            return False
        if wall.decay_rate < -params.max_negative_decay:
            return False
        if wall.touch_count < params.min_touch_count:
            return False
        if wall.health_state not in params.allowed_health_states:
            return False
        if wall.health_change < -params.max_negative_health_change:
            return False
        if wall.health_velocity < -params.max_negative_health_velocity:
            return False
        return True
