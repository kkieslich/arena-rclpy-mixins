from __future__ import annotations

import threading
import time

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
