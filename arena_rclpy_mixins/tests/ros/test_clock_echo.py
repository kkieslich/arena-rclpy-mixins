from __future__ import annotations

import time

import rclpy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


class _Rig:
    def __init__(self, name: str, min_interval_s: float) -> None:
        from arena_rclpy_mixins.Time import ClockEcho, TimeNode

        self.node = TimeNode(name)
        self.clock = self.node.create_publisher(Clock, "/clock", 10)
        self.echo: ClockEcho[String] = ClockEcho(self.node, self.node.create_publisher(String, f"/{name}/out", 10), min_interval_s=min_interval_s)
        self.seen: list[str] = []
        self.ticks: list[float] = []
        self.node.create_subscription(String, f"/{name}/out", lambda msg: self.seen.append(msg.data), 10)
        self.node.on_clock(lambda now: self.ticks.append(now.to_seconds()))
        self._settle(lambda: self.node.count_subscribers("/clock") > 0 and self.echo.publisher.get_subscription_count() > 0)

    def _settle(self, done, seconds: float = 10.0) -> None:
        deadline = time.monotonic() + seconds
        while not done():
            assert time.monotonic() < deadline, "the rig never settled"
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def publish(self, data: str) -> None:
        count = len(self.seen)
        self.echo.publish(String(data=data))
        self._settle(lambda: len(self.seen) > count)

    def tick(self, seconds: float) -> None:
        from arena_rclpy_mixins.Time import Time

        count = len(self.ticks)
        self.clock.publish(Clock(clock=Time.from_float(seconds).to_msg()))
        self._settle(lambda: len(self.ticks) > count)
        end = time.monotonic() + 0.1
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def close(self) -> None:
        self.echo.close()
        self.node.destroy_node()


def test_clock_echo_resends_the_latest_message_on_an_exact_clock_repeat_only() -> None:
    rig = _Rig("clock_echo_repeat", min_interval_s=0.0)
    try:
        rig.publish("a")
        rig.tick(1.0)
        rig.tick(1.0)
        rig.publish("b")
        rig.tick(2.0)
        rig.tick(2.0)
        assert rig.seen == ["a", "a", "b", "b"]
    finally:
        rig.close()


def test_clock_echo_drops_its_message_when_the_clock_goes_back() -> None:
    rig = _Rig("clock_echo_back", min_interval_s=0.0)
    try:
        rig.publish("a")
        rig.tick(2.0)
        rig.tick(1.0)
        rig.tick(1.0)
        rig.publish("b")
        rig.tick(1.0)
        assert rig.seen == ["a", "b", "b"]
    finally:
        rig.close()


def test_clock_echo_waits_out_its_interval_between_echoes() -> None:
    rig = _Rig("clock_echo_interval", min_interval_s=60.0)
    try:
        rig.publish("a")
        rig.tick(1.0)
        rig.tick(1.0)
        rig.tick(1.0)
        assert rig.seen == ["a", "a"]
    finally:
        rig.close()


def test_clock_echo_stays_silent_before_its_first_message_and_after_close() -> None:
    rig = _Rig("clock_echo_silent", min_interval_s=0.0)
    try:
        rig.tick(1.0)
        rig.tick(1.0)
        assert rig.seen == []
        rig.publish("a")
        rig.echo.close()
        rig.tick(1.0)
        assert rig.seen == ["a"]
    finally:
        rig.node.destroy_node()
