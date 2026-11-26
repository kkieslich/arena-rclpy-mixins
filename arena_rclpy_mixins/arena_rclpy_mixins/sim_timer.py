"""A timer on a node's sim clock that the node's /clock messages fire, instead of an rcl timer on the ROS clock."""

from __future__ import annotations

import inspect
import threading
import typing

import rclpy.callback_groups
import rclpy.clock
import rclpy.node
import rclpy.timer

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.Time import Time, TimeNode

_NANOSECONDS_PER_SECOND = 10**9
#: the steady kick that hands a due call to the executor fires this soon after it is armed [ns]
_KICK_NS = 1_000
#: and then not again for this long, unless it is armed again [ns]: one expiry per arming, even on a lagging executor
_KICK_IDLE_NS = 3600 * _NANOSECONDS_PER_SECOND


class SimClockTimer:
    """The subset of `rclpy.timer.Timer` a TimeNode's users rely on, for a timer on the node's sim clock.

    An rcl timer on the ROS clock is woken by the EventsExecutor on every clock jump and timer reset. One that fell behind
    was dispatched over and over while its next call was periods away, and that storm held the executor's queue for tens
    of seconds on a loaded node. This timer has no rcl timer on the ROS clock. The node's /clock listener checks it on every
    message. Once sim time reaches its next call, it arms a steady one-shot rcl timer in the timer's callback group, which
    hands the callback to the node's executor (the EventsExecutor takes no guard conditions). While that dispatch waits,
    no second one is armed. When the callback runs, the next call moves on as
    `rcl_timer_call` moves it: by a period, and past every mark the clock already passed (missed calls are dropped, not
    caught up). A backward clock jump starts the period over from the new time, as rcl does.
    """

    def __init__(
        self,
        node: TimeNode,
        period_sec: float,
        callback: typing.Callable[[], typing.Any],
        callback_group: rclpy.callback_groups.CallbackGroup | None,
        *,
        autostart: bool = True,
    ) -> None:
        self._node = node
        self.callback = callback
        self.callback_group = callback_group
        self.clock = node.get_clock()
        self._period_ns = int(float(period_sec) * _NANOSECONDS_PER_SECOND)
        self._lock = threading.Lock()
        self._canceled = not autostart
        self._pending = False
        self._destroyed = False
        now = self._now_ns()
        self._seen_ns: int | None = now
        self._next_ns: int | None = None if now is None else now + self._period_ns
        self._last_ns: int | None = now
        steady = rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        dispatch = self._dispatch_async if inspect.iscoroutinefunction(callback) else self._dispatch
        self._kick: rclpy.timer.Timer = rclpy.node.Node.create_timer(node, _KICK_IDLE_NS / _NANOSECONDS_PER_SECOND, dispatch, callback_group=callback_group, clock=steady, autostart=False)
        self._remove_listener = node.on_clock(self._on_clock)

    def _now_ns(self) -> int | None:
        """The node's sim time, None before its first /clock message."""
        sim = self._node.sim_time.to_nanoseconds()
        clock = self.clock
        if sim == 0 and not (isinstance(clock, rclpy.clock.ROSClock) and clock.ros_time_is_active):
            return None
        return sim

    def _on_clock(self, now: Time) -> None:
        del now
        now_ns = self._now_ns()
        if now_ns is None:
            return
        with self._lock:
            if self._destroyed:
                return
            seen, self._seen_ns = self._seen_ns, now_ns
            if self._next_ns is None or self._last_ns is None:
                # the first clock: the timer starts now, as an rcl timer does when ROS time is activated
                self._next_ns, self._last_ns = now_ns + self._period_ns, now_ns
                return
            if seen is not None and now_ns < seen:
                self._next_ns, self._last_ns = now_ns + self._period_ns, now_ns
            due = not self._canceled and not self._pending and now_ns >= self._next_ns
            if due:
                self._pending = True
        if due:
            self._arm_kick()

    def _arm_kick(self) -> None:
        """Make the kick expire once, now: a short period for the reset, then the idle period again, so the rcl timer is not
        due again (and the executor not woken again) while its one dispatch waits."""
        kick = self._kick
        kick.timer_period_ns = _KICK_NS
        kick.reset()
        kick.timer_period_ns = _KICK_IDLE_NS

    def _take(self) -> bool:
        """Whether the call the kick was armed for is still due, moving the next call on if it is."""
        self._kick.cancel()
        with self._lock:
            self._pending = False
            now_ns = self._now_ns()
            if self._destroyed or self._canceled or now_ns is None or self._next_ns is None or now_ns < self._next_ns:
                return False
            self._advance(now_ns)
            return True

    def _dispatch(self) -> typing.Any:  # noqa: ANN401
        return self.callback() if self._take() else None

    async def _dispatch_async(self) -> typing.Any:  # noqa: ANN401
        return await self.callback() if self._take() else None

    def _advance(self, now_ns: int) -> None:
        """Move the next call on as `rcl_timer_call` does, the lock held."""
        assert self._next_ns is not None
        period = self._period_ns
        self._last_ns = now_ns
        next_ns = self._next_ns + period
        if next_ns < now_ns:
            if period == 0:
                next_ns = now_ns
            else:
                next_ns += ((now_ns - next_ns) // period + 1) * period
        self._next_ns = next_ns

    @property
    def timer_period_ns(self) -> int:
        return self._period_ns

    @timer_period_ns.setter
    def timer_period_ns(self, value: int) -> None:
        with self._lock:
            self._period_ns = int(value)

    def time_until_next_call(self) -> int | None:
        with self._lock:
            if self._canceled or self._destroyed:
                return None
            now_ns = self._now_ns()
            if now_ns is None or self._next_ns is None:
                return self._period_ns
            return self._next_ns - now_ns

    def time_since_last_call(self) -> int:
        with self._lock:
            now_ns = self._now_ns()
            if now_ns is None or self._last_ns is None:
                return 0
            return now_ns - self._last_ns

    def is_ready(self) -> bool:
        with self._lock:
            now_ns = self._now_ns()
            return not self._canceled and now_ns is not None and self._next_ns is not None and now_ns >= self._next_ns

    def is_canceled(self) -> bool:
        return self._canceled

    def cancel(self) -> None:
        with self._lock:
            self._canceled = True

    def reset(self) -> None:
        """Start the period over from now (and resume a cancelled timer), as `rcl_timer_reset`."""
        with self._lock:
            self._canceled = False
            now_ns = self._now_ns()
            if now_ns is not None:
                self._next_ns, self._last_ns = now_ns + self._period_ns, now_ns

    def destroy(self) -> None:
        with self._lock:
            if self._destroyed:
                return
            self._destroyed = True
        self._remove_listener()
        rclpy.node.Node.destroy_timer(self._node, self._kick)
