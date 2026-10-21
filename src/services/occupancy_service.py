"""Occupancy-session lifecycle on the video timeline.

Processing-run policy
---------------------
A processing run is identified by the background processing thread's
lifecycle -- never by video timestamps. Every run records its own session
rows through ``database.repository`` and never updates rows of a previous
run:

- At the start of a run, every session left active by an interrupted run
  (e.g. after a crash) is closed at its own start time, so old rows can
  never absorb or extend the new run.
- At the end of a run, every session still active is closed at the last
  processed video timestamp.
- Browser connects, refreshes, and disconnects never start runs and never
  touch sessions.

Invalid timestamps are prevented in the repository (``ended_at`` is clamped
to ``started_at``), and the partial unique index on active sessions remains
the hard guarantee that only one session per table can be active.
"""

from database.repository import (
    close_all_sessions,
    close_leftover_sessions,
    close_session,
    start_session,
    update_session_max,
)


class OccupancyService:
    """Manage per-table occupancy sessions for one processing run.

    ``update`` must be called with the stabilized table count of every
    frame; database writes happen only on session transitions (start, new
    maximum, end) -- never on ordinary frames.
    """

    def __init__(self, table_names: list[str]):
        self._start: dict[str, float | None] = {name: None for name in table_names}
        self._max_people: dict[str, int] = {name: 0 for name in table_names}

    def begin_run(self) -> None:
        """Start a processing run: clear leftovers and in-memory state.

        Sessions left active by an interrupted previous run are closed first
        (history is preserved, ``ended_at`` is never earlier than
        ``started_at``), so only this run's sessions can be active.
        """
        close_leftover_sessions()
        for name in self._start:
            self._start[name] = None
            self._max_people[name] = 0

    def update(self, name: str, count: int, video_time: float) -> float | None:
        """Record one frame's stabilized count; return elapsed seconds or None.

        A session starts at the 0 -> positive transition, keeps running
        while the stabilized count stays positive (a temporary miss cannot
        reset an active visit), and closes when the count returns to zero.
        """
        start = self._start[name]

        if count > 0:
            if start is None:
                start = self._start[name] = video_time
                self._max_people[name] = count
                start_session(name, video_time, count)
            elif count > self._max_people[name]:
                self._max_people[name] = count
                update_session_max(name, count)
            return video_time - start

        if start is not None:
            close_session(name, video_time)
        self._start[name] = None
        self._max_people[name] = 0
        return None

    def end_run(self, last_video_time: float) -> None:
        """Close every session still active when processing ends.

        Called with the last processed video timestamp, so sessions of
        people still seated at the end of the video are closed there and
        never left open (the repository clamps end times to start times).
        """
        close_all_sessions(last_video_time)
        self.begin_run()
