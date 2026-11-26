"""SimClockTimer: the same sim marks as an rcl timer on the ROS clock for scripted /clock streams, dispatch through the
executor (coroutines, callback groups), no storm under lag, and rclpy's own timer without sim time."""

from __future__ import annotations

import threading
import time
import typing

import pytest
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.parameter
import rclpy.time
from rosgraph_msgs.msg import Clock

PERIOD_S = 0.1
MS = 1_000_000


def _sim(name: str) -> list[rclpy.parameter.Parameter]:
    del name
    return [rclpy.parameter.Parameter("use_sim_time", value=True)]


class _Pair:
    """A TimeNode (SimClockTimer) and a plain rclpy node (rcl timer on the ROS clock), each on its own SingleThreadedExecutor, fed the same /clock."""

    def __init__(self, tag: str) -> None:
        from arena_rclpy_mixins.Time import TimeNode

        self.ours = TimeNode(f"sim_clock_ours_{tag}", parameter_overrides=_sim(tag))
        self.rcl = rclpy.node.Node(f"sim_clock_rcl_{tag}", parameter_overrides=_sim(tag))
        self.driver = rclpy.node.Node(f"sim_clock_driver_{tag}")
        self.clock = self.driver.create_publisher(Clock, "/clock", 10)
        self.executors = [rclpy.executors.SingleThreadedExecutor(), rclpy.executors.SingleThreadedExecutor()]
        self.executors[0].add_node(self.ours)
        self.executors[1].add_node(self.rcl)
        self.fired: dict[str, list[int]] = {"ours": [], "rcl": []}
        deadline = time.monotonic() + 10.0
        while self.driver.count_subscribers("/clock") < 3:
            assert time.monotonic() < deadline, "the nodes never subscribed to /clock"
            time.sleep(0.02)

    def timers(self, period_s: float = PERIOD_S) -> tuple[typing.Any, typing.Any]:
        ours = self.ours.create_timer(period_s, lambda: self.fired["ours"].append(self.ours.get_clock().now().nanoseconds))
        rcl = self.rcl.create_timer(period_s, lambda: self.fired["rcl"].append(self.rcl.get_clock().now().nanoseconds))
        return ours, rcl

    def step(self, ns: int, spins: int = 6) -> None:
        """Publish one /clock and spin both nodes until they hold it, then let their timers run."""
        self.clock.publish(Clock(clock=rclpy.time.Time(nanoseconds=ns).to_msg()))
        deadline = time.monotonic() + 5.0
        while self.ours.sim_time.to_nanoseconds() != ns or self.rcl.get_clock().now().nanoseconds != ns:
            assert time.monotonic() < deadline, f"the nodes never saw the clock at {ns}"
            for executor in self.executors:
                executor.spin_once(timeout_sec=0.005)
        for _ in range(spins):
            for executor in self.executors:
                executor.spin_once(timeout_sec=0.002)

    def close(self) -> None:
        for executor in self.executors:
            executor.shutdown()
        for node in (self.ours, self.rcl, self.driver):
            node.destroy_node()


def _script(kind: str) -> list[int]:
    """A /clock stream in ms: steady 10 ms steps, with the event `kind` in the middle."""
    steps = list(range(1000, 1600, 10))
    if kind == "jump":
        return steps + [t + 3500 for t in range(1600, 2000, 10)]
    if kind == "repeat":
        return [t for t in steps for _ in range(3 if t % 70 == 0 else 1)]
    if kind == "backward":
        return steps + [t - 400 for t in range(1600, 2000, 10)]
    if kind == "pause":
        return steps + [1590] * 30 + list(range(1600, 1900, 10))
    if kind == "uneven":
        return [1000 + (k * k * 7) % 997 // 3 + 10 * k for k in range(80)]
    return steps


@pytest.mark.parametrize("kind", ["steady", "jump", "repeat", "backward", "pause", "uneven"])
def test_a_sim_clock_timer_fires_on_the_marks_of_an_rcl_timer(kind: str) -> None:
    pair = _Pair(kind)
    try:
        pair.step(1000 * MS)
        pair.timers()
        stream = sorted(set(_script(kind))) if kind == "uneven" else _script(kind)
        for ms in stream:
            pair.step(ms * MS)
        assert pair.fired["ours"] == pair.fired["rcl"]
        assert len(pair.fired["ours"]) >= 4
    finally:
        pair.close()


@pytest.mark.parametrize("action", ["period", "cancel_reset", "one_shot", "until"])
def test_period_change_cancel_reset_one_shot_and_time_until_next_call_follow_rcl(action: str) -> None:
    pair = _Pair(action)
    try:
        pair.step(1000 * MS)
        ours, rcl = pair.timers()
        alive = True
        untils: list[tuple[int | None, int | None]] = []
        for k, ms in enumerate(range(1000, 2500, 10)):
            if action == "period" and k == 40:
                ours.timer_period_ns = rcl.timer_period_ns = 250 * MS
            if action == "cancel_reset" and k == 30:
                ours.cancel()
                rcl.cancel()
            if action == "cancel_reset" and k == 70:
                ours.reset()
                rcl.reset()
            if action == "one_shot" and pair.fired["ours"] and pair.fired["rcl"] and alive:
                assert pair.ours.destroy_timer(ours)
                _ = pair.rcl.destroy_timer(rcl)
                alive = False
            pair.step(ms * MS)
            if action == "until":
                untils.append((ours.time_until_next_call(), rcl.time_until_next_call()))
        assert pair.fired["ours"] == pair.fired["rcl"]
        if action == "one_shot":
            assert len(pair.fired["ours"]) == 1
        if action == "cancel_reset":
            assert ours.is_canceled() == rcl.is_canceled() is False
        if action == "until":
            assert untils == [(a, a) for a, _ in untils] and [a for a, _ in untils] == [b for _, b in untils]
    finally:
        pair.close()


def test_without_sim_time_or_on_another_clock_the_timer_is_rclpys() -> None:
    import rclpy.clock
    import rclpy.timer

    from arena_rclpy_mixins.sim_timer import SimClockTimer
    from arena_rclpy_mixins.Time import TimeNode

    wall = TimeNode("sim_clock_wall")
    sim = TimeNode("sim_clock_sim", parameter_overrides=_sim("sim"))
    try:
        assert type(wall.create_timer(0.1, lambda: None)) is rclpy.timer.Timer
        steady = rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        assert type(sim.create_timer(0.1, lambda: None, clock=steady)) is rclpy.timer.Timer
        assert type(sim.wall_timer(0.1, lambda: None)) is rclpy.timer.Timer
        assert isinstance(sim.create_timer(0.1, lambda: None), SimClockTimer)
    finally:
        wall.destroy_node()
        sim.destroy_node()


@pytest.mark.parametrize("kind", ["events", "single"])
def test_coroutines_run_through_the_executor_and_a_callback_group_stays_exclusive(kind: str) -> None:
    from arena_rclpy_mixins.spin import create_executor
    from arena_rclpy_mixins.Time import Time, TimeNode

    node = TimeNode(f"sim_clock_async_{kind}", parameter_overrides=_sim(kind))
    driver = rclpy.node.Node(f"sim_clock_async_driver_{kind}")
    clock = driver.create_publisher(Clock, "/clock", 10)
    executor = create_executor() if kind == "events" else rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
    inside = [0]
    overlaps: list[int] = []
    ran: list[str] = []

    async def slow() -> None:
        inside[0] += 1
        overlaps.append(inside[0])
        ran.append(threading.current_thread().name)
        time.sleep(0.03)
        inside[0] -= 1

    def fast() -> None:
        inside[0] += 1
        overlaps.append(inside[0])
        time.sleep(0.03)
        inside[0] -= 1

    node.create_timer(0.05, slow, callback_group=group)
    node.create_timer(0.05, fast, callback_group=group)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        deadline = time.monotonic() + 10.0
        while driver.count_subscribers("/clock") < 2:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        now = 1.0
        for _ in range(60):
            now += 0.01
            clock.publish(Clock(clock=Time.from_float(now).to_msg()))
            time.sleep(0.01)
        assert len(ran) >= 3, "the coroutine timer never ran"
        assert max(overlaps) == 1, "two callbacks of one mutually exclusive group ran at once"
    finally:
        executor.shutdown()
        spinner.join(timeout=5.0)
        driver.destroy_node()
        node.destroy_node()


def test_a_lagging_events_executor_dispatches_each_sim_timer_at_most_once_per_mark() -> None:
    """The storm this timer replaces: an EventsExecutor held in long callbacks while /clock runs on, and the interpreter
    held for over a second twice; every timer's callbacks stay within one per 0.1 s mark of sim time, and its kick is
    dispatched at most twice per call (it does not expire again while its dispatch waits)."""
    from std_msgs.msg import Empty

    from arena_rclpy_mixins.spin import create_executor
    from arena_rclpy_mixins.Time import Time, TimeNode

    node = TimeNode("sim_clock_lag", parameter_overrides=_sim("lag"))
    driver = rclpy.node.Node("sim_clock_lag_driver")
    clock = driver.create_publisher(Clock, "/clock", 10)
    load = driver.create_publisher(Empty, "/sim_clock_lag_load", 1000)
    from arena_rclpy_mixins import sim_timer

    runs = [0] * 4
    takes = [0]
    take = sim_timer.SimClockTimer._take

    def counted(self: typing.Any) -> bool:
        takes[0] += 1
        return take(self)

    sim_timer.SimClockTimer._take = counted  # pyright: ignore[reportAttributeAccessIssue]
    for i in range(4):
        node.create_timer(PERIOD_S, lambda i=i: runs.__setitem__(i, runs[i] + 1))
    node.create_subscription(Empty, "/sim_clock_lag_load", lambda _m: time.sleep(0.004), 1000)
    beats: list[float] = []
    node.wall_timer(0.5, lambda: beats.append(time.monotonic()))
    executor = create_executor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        deadline = time.monotonic() + 10.0
        while driver.count_subscribers("/clock") < 2 or driver.count_subscribers(load.topic_name) < 1:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        start, now = time.monotonic(), 1.0
        held: dict[int, bool] = {}
        while time.monotonic() - start < 6.0:
            now += 1 / 30
            clock.publish(Clock(clock=Time.from_float(now).to_msg()))
            for _ in range(10):  # 300 load messages/s at 4 ms each: the executor runs behind
                load.publish(Empty())
            if int(time.monotonic() - start) in (2, 4) and not held.get(int(time.monotonic() - start)):
                held[int(time.monotonic() - start)] = True
                end = time.monotonic() + 1.2  # this thread holds the interpreter for over a second, several timers fall due
                while time.monotonic() < end:
                    pass
            time.sleep(1 / 30)
        marks = int((now - 1.0) / PERIOD_S) + 1
        assert all(r <= marks for r in runs), f"a timer ran {max(runs)} times for {marks} marks"
        assert sum(runs) > 0 and takes[0] <= 2 * sum(runs), f"{takes[0]} kick dispatches for {sum(runs)} calls"
    finally:
        sim_timer.SimClockTimer._take = take  # pyright: ignore[reportAttributeAccessIssue]
        executor.shutdown()
        spinner.join(timeout=5.0)
        driver.destroy_node()
        node.destroy_node()
