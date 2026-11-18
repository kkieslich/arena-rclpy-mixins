"""wait_for_lifecycle_state_async keeps polling past a get_state reply that never comes."""

from __future__ import annotations

import asyncio
import threading

import rclpy
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from rclpy.callback_groups import ReentrantCallbackGroup


class HeldFirstReply:
    """A get_state server that holds its first reply until released and answers every later request as active."""

    def __init__(self, name: str, release: threading.Event) -> None:
        self.node = rclpy.create_node(f"{name}_server")
        self.requests = 0
        self._release = release
        self.node.create_service(GetState, f"/{name}/get_state", self._answer, callback_group=ReentrantCallbackGroup())

    def _answer(self, request, response):
        self.requests += 1
        if self.requests == 1:
            self._release.wait(60.0)
        response.current_state.id = State.PRIMARY_STATE_ACTIVE
        return response


def test_a_held_get_state_reply_costs_one_attempt_not_the_whole_wait() -> None:
    from arena_rclpy_mixins.LifecycleClient import AsyncLifecycleClient

    async def main() -> tuple[bool, int]:
        release = threading.Event()
        server = HeldFirstReply("lifecycle_wait_held", release)
        node = AsyncLifecycleClient("lifecycle_wait_held_client")
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        executor.add_node(server.node)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            active = await node.wait_for_lifecycle_state_async("/lifecycle_wait_held", State.PRIMARY_STATE_ACTIVE, timeout=20.0)
            return active, server.requests
        finally:
            release.set()
            executor.shutdown()
            thread.join(timeout=5.0)
            node.destroy_node()
            server.node.destroy_node()

    active, requests = asyncio.run(main())
    assert active
    assert requests >= 2
