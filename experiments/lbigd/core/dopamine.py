from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DopamineGate:
    """
    Event gate driven by prediction error / surprise.

    - Maintains an EMA baseline of error
    - Opens when current error spikes above baseline * ratio
    - Holds open for a few steps (phasic burst proxy)
    """

    ratio: float = 1.8
    ema_decay: float = 0.98
    hold_steps: int = 10
    min_threshold: float = 0.0

    _ema: float | None = None
    _hold: int = 0

    def update(self, error: float) -> bool:
        if error < 0:
            raise ValueError("error must be >= 0")

        if self._hold > 0:
            self._hold -= 1
            self._update_ema(error)
            return True

        self._update_ema(error)
        threshold = max(float(self.min_threshold), float(self._ema or 0.0) * float(self.ratio))
        if error > threshold:
            self._hold = max(0, int(self.hold_steps) - 1)
            return True
        return False

    def baseline(self) -> float:
        return float(self._ema or 0.0)

    def _update_ema(self, error: float) -> None:
        if self._ema is None:
            self._ema = float(error)
            return
        d = float(self.ema_decay)
        self._ema = d * float(self._ema) + (1.0 - d) * float(error)


