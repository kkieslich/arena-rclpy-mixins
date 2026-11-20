from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest
import rclpy.executors
import rclpy.node
import rclpy.parameter
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

EXECUTORS = ("events", "single")
PERIOD_NS = 100_000_000
STEP_NS = 10_000_000
GATE_NS = 110_000_000
GATES = 1500
LOAD_S = 0.001
FLOOD_S = 0.002


def _executor(kind: str) -> rclpy.executors.Executor:
    from arena_rclpy_mixins.spin import create_executor

    return {"events": create_executor, "single": rclpy.executors.SingleThreadedExecutor}[kind]()


@pytest.mark.parametrize("kind", EXECUTORS)
def test_sim_clock_timer_beats_once_per_period_on_its_grid_through_lockstep_gates_under_load(kind: str) -> None:
    from arena_rclpy_mixins.Time import Time, TimeNode

    node = TimeNode(f"sim_timer_{kind}", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
    driver = rclpy.node.Node(f"sim_timer_driver_{kind}")
    clock = driver.create_publisher(Clock, "/clock", 10)
    load = driver.create_publisher(String, f"/sim_timer_load_{kind}", 100)
    beats: list[int] = []
    calls: list[int] = []
    held: list[int] = []
    beat = threading.Condition()

    def _beat() -> None:
        with beat:
            beats.append(node.get_clock().now().nanoseconds)
            held.append(node.sim_time.to_nanoseconds())
            calls.append(beats[-1] + (timer.time_until_next_call() or 0))
            beat.notify_all()

    timer = node.create_timer(PERIOD_NS * 1e-9, _beat)
    node.create_subscription(String, f"/sim_timer_load_{kind}", lambda _: time.sleep(LOAD_S), 100)
    executor = _executor(kind)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    stop = threading.Event()

    def _flood() -> None:
        while not stop.is_set():
            load.publish(String(data="load"))
            time.sleep(FLOOD_S)

    flooder = threading.Thread(target=_flood, daemon=True)
    try:
        deadline = time.monotonic() + 10.0
        while driver.count_subscribers("/clock") < 2 or driver.count_subscribers(load.topic_name) < 1:
            assert time.monotonic() < deadline, "the driver never matched the node"
            time.sleep(0.02)
        flooder.start()
        now = 0
        clock.publish(Clock(clock=Time().to_msg()))
        for gate in range(GATES):
            with beat:
                due = (beats[-1] if beats else 0) + GATE_NS
            while now < due:
                now += STEP_NS
                clock.publish(Clock(clock=Time.from_float(now * 1e-9).to_msg()))
                time.sleep(0.0002)
            with beat:
                covered = beat.wait_for(lambda now=now: bool(beats) and beats[-1] > now - GATE_NS, timeout=5.0)
            assert covered, f"gate {gate} at {now * 1e-9:.2f} s never covered, last beat at {beats[-1] * 1e-9 if beats else None} s"
        assert len(beats) == len(set(beats)), "a beat repeated its clock time"
        assert [(now, sim) for now, sim in zip(beats, held, strict=True) if sim < now] == [], "sim_time fell behind the clock that fired the timer"
        assert {call % PERIOD_NS for call in calls} == {calls[0] % PERIOD_NS}, "the timer left its period grid"
    finally:
        stop.set()
        executor.shutdown()
        spinner.join(timeout=5.0)
        driver.destroy_node()
        node.destroy_node()


def test_service_keeps_answering_while_a_sim_clock_timer_outruns_its_period() -> None:
    from std_srvs.srv import Trigger

    from arena_rclpy_mixins.spin import create_executor
    from arena_rclpy_mixins.Time import Time, TimeNode

    node = TimeNode("sim_timer_overrun", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
    driver = rclpy.node.Node("sim_timer_overrun_driver")
    clock = driver.create_publisher(Clock, "/clock", 10)
    client = driver.create_client(Trigger, "/sim_timer_overrun/ping")
    node.create_timer(0.05, lambda: time.sleep(0.3))
    for _ in range(3):
        node.create_timer(0.05, lambda: None)
    node.create_service(Trigger, "/sim_timer_overrun/ping", lambda _req, res: res)
    executor = create_executor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    driver_executor = rclpy.executors.SingleThreadedExecutor()
    driver_executor.add_node(driver)
    stop = threading.Event()

    def _tick() -> None:
        now = 1_000_000_000
        while not stop.is_set():
            clock.publish(Clock(clock=Time.from_float(now * 1e-9).to_msg()))
            now += STEP_NS
            time.sleep(STEP_NS * 1e-9)

    ticker = threading.Thread(target=_tick, daemon=True)
    try:
        assert client.wait_for_service(timeout_sec=10.0), "the node never offered its service"
        ticker.start()
        cpu, wall = time.process_time(), time.monotonic()
        latencies = []
        for _ in range(20):
            start = time.monotonic()
            future = client.call_async(Trigger.Request())
            driver_executor.spin_until_future_complete(future, timeout_sec=5.0)
            assert future.done(), "the service stopped answering"
            latencies.append(time.monotonic() - start)
            time.sleep(0.2)
        assert (time.process_time() - cpu) / (time.monotonic() - wall) < 0.2, "repeated timer dispatches kept the process busy"
        assert max(latencies) < 2.0, f"the slowest answer took {max(latencies):.2f} s"
    finally:
        stop.set()
        executor.shutdown()
        spinner.join(timeout=5.0)
        driver.destroy_node()
        node.destroy_node()


@pytest.mark.parametrize("action", ["none", "cancel", "retune", "reset", "destroy", "late"])
def test_skipped_dispatch_is_moved_back_only_while_the_caller_left_the_timer_alone(action: str) -> None:
    from arena_rclpy_mixins.Time import Time, TimeNode

    node = TimeNode(f"sim_timer_owed_{action}", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
    driver = rclpy.node.Node(f"sim_timer_owed_driver_{action}")
    clock = driver.create_publisher(Clock, "/clock", 10)
    beats: list[int] = []

    def _advance(now_ns: int) -> None:
        deadline = time.monotonic() + 10.0
        while node.sim_time.to_nanoseconds() != now_ns:
            assert time.monotonic() < deadline, "the node never saw the clock"
            clock.publish(Clock(clock=Time.from_float(now_ns * 1e-9).to_msg()))
            rclpy.spin_once(node, timeout_sec=0.02)

    try:
        start = 1_000_000_000
        _advance(start)
        timer = node.create_timer(PERIOD_NS * 1e-9, lambda: beats.append(node.sim_time.to_nanoseconds()))
        timer.timer_period_ns = 3 * PERIOD_NS
        timer.reset()
        timer.timer_period_ns = PERIOD_NS
        timer.callback()
        assert beats == [], "a dispatch three periods early ran the callback"
        if action == "cancel":
            timer.cancel()
        elif action == "retune":
            timer.timer_period_ns = 2 * PERIOD_NS
        elif action == "reset":
            timer.reset()
        elif action == "destroy":
            node.destroy_timer(timer)
            destroyed = weakref.ref(timer)
            del timer
            gc.collect()
            assert destroyed() is None, "the node kept a destroyed timer alive"
            return
        if action == "late":
            _advance(start + 2 * PERIOD_NS + PERIOD_NS // 2)
            deadline = time.monotonic() + 10.0
            while not beats:
                assert time.monotonic() < deadline, "the tick the repeated dispatch took never fired"
                rclpy.spin_once(node, timeout_sec=0.02)
            assert beats == [start + 2 * PERIOD_NS + PERIOD_NS // 2]
        else:
            _advance(start + STEP_NS)

        def _next_call() -> int:
            return node.get_clock().now().nanoseconds + (timer.time_until_next_call() or 0)

        if action == "none":
            assert _next_call() == start + 2 * PERIOD_NS
        elif action == "cancel":
            assert timer.is_canceled()
        elif action == "retune":
            assert timer.timer_period_ns == 2 * PERIOD_NS
            assert _next_call() == start + 3 * PERIOD_NS
        elif action == "reset":
            assert _next_call() == start + PERIOD_NS
        elif action == "late":
            assert _next_call() == start + 3 * PERIOD_NS
    finally:
        driver.destroy_node()
        node.destroy_node()
