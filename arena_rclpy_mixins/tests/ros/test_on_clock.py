from __future__ import annotations

import time

import rclpy
from rosgraph_msgs.msg import Clock


def _publish_until(node, publisher, seconds: float, done) -> None:
    from arena_rclpy_mixins.Time import Time

    deadline = time.monotonic() + 10.0
    while not done():
        assert time.monotonic() < deadline, "the clock message never reached the node"
        publisher.publish(Clock(clock=Time.from_float(seconds).to_msg()))
        rclpy.spin_once(node, timeout_sec=0.02)


def test_on_clock_sees_each_clock_time_after_sim_time_holds_it_until_removed() -> None:
    from arena_rclpy_mixins.Time import Time, TimeNode

    node = TimeNode("on_clock_listener")
    publisher = node.create_publisher(Clock, "/clock", 10)
    seen: list[tuple[Time, Time]] = []
    try:
        remove = node.on_clock(lambda now: seen.append((now, node.sim_time)))
        _publish_until(node, publisher, 3.5, lambda: any(now == Time.from_float(3.5) for now, _ in seen))
        assert all(now == held for now, held in seen)

        remove()
        count = len(seen)
        _publish_until(node, publisher, 4.5, lambda: node.sim_time == Time.from_float(4.5))
        assert len(seen) == count
    finally:
        node.destroy_publisher(publisher)
        node.destroy_node()
