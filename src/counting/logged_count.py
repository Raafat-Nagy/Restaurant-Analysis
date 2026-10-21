"""Stabilize per-table person counts against per-frame detection noise."""

from collections import deque


class LoggedCount:
    """Stabilize one table's person count from noisy per-frame readings.

    The latest raw readings are kept in ``history``
    (``collections.deque(maxlen=...)``, 10 by default). The candidate
    count is the majority vote of that history: the value that appears
    most frequently, with ties preferring the stable count. One or two
    noisy frames (for example a seated person briefly missed by the
    detector) cannot win the vote against the rest of the window.

    A new candidate becomes the stable count only after it stays the
    majority winner for ``increase_seconds`` (count went up) or
    ``decrease_seconds`` (count went down). Elapsed time is measured with
    the ``now`` value passed to ``update`` — the video's own timeline in
    seconds for prerecorded footage — never with frame counts or with
    wall-clock processing time, and the confirmation window always runs
    the full configured delay. An isolated fluctuation
    cannot restart the timer: the vote is re-evaluated every reading and
    the timer only restarts when the majority winner actually changes.

    A table is reported empty only after ``empty_seconds`` of continuous
    zero readings, so one zero frame never resets the count and old
    nonzero readings in the history never delay the reset. Once a
    transition is confirmed, the pending candidate state and the history
    are reset, and the stable count can decrease again later (the highest
    count is never kept permanently).

    One instance manages one table; create one instance per table so every
    table's count is stabilized independently.
    """

    def __init__(
        self,
        increase_seconds: float = 2.0,
        decrease_seconds: float = 10.0,
        history_size: int = 10,
        empty_seconds: float = 10.0,
    ):
        self.increase_seconds = increase_seconds
        self.decrease_seconds = decrease_seconds
        self.empty_seconds = empty_seconds
        self.history = deque(maxlen=history_size)
        self.stable = 0
        self._candidate = None
        self._candidate_since = None
        self._empty_since = None

    def update(self, raw: int, now: float) -> int:
        """Log one raw reading at video time ``now`` and return the stabilized count."""
        raw = max(0, int(raw))
        self.history.append(raw)

        # Empty tables are confirmed by time alone: continuous zero
        # readings for ``empty_seconds``. One zero frame never resets the
        # count, and old nonzero values in the history never delay it.
        if self.stable > 0 and raw == 0:
            if self._empty_since is None:
                self._empty_since = now
            if now - self._empty_since + 1e-9 >= self.empty_seconds:
                self.stable = 0
                self._reset_state()
                return self.stable
        else:
            self._empty_since = None

        # Majority vote over the recent history. Drops to zero are owned
        # by the empty check above.
        candidate = self._majority()
        if candidate == 0 and self.stable > 0:
            candidate = self.stable

        if candidate == self.stable:
            # The history agrees with the stable count: no change pending.
            self._clear_candidate()
            return self.stable

        if self._candidate != candidate:
            # New majority winner: start its full confirmation window.
            self._candidate = candidate
            self._candidate_since = now
            return self.stable

        # Same winner: keep its timer running. An isolated raw
        # fluctuation cannot restart it because the vote above still
        # supports this candidate.
        required = self.increase_seconds if candidate > self.stable else self.decrease_seconds
        held = now - self._candidate_since
        if held + 1e-9 < required:  # tolerate float rounding of timestamps
            return self.stable

        self.stable = candidate
        self._clear_candidate()
        self.history.clear()
        return self.stable

    def _majority(self) -> int:
        """Most frequent recent reading, as a non-negative integer.

        Ties prefer the current stable count so alternating detections
        (e.g. 2, 1, 2, 1) never propose a change. If the stable count is
        not among the tied values, the lowest tied value is chosen so the
        result is always deterministic.
        """
        frequencies = {}
        for value in self.history:
            frequencies[value] = frequencies.get(value, 0) + 1
        top = max(frequencies.values())
        tied = [value for value, count in frequencies.items() if count == top]
        if self.stable in tied:
            return self.stable
        return min(tied)

    def _clear_candidate(self):
        self._candidate = None
        self._candidate_since = None

    def _reset_state(self):
        self._clear_candidate()
        self._empty_since = None
        self.history.clear()
