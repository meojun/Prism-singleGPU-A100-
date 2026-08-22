#!/usr/bin/env python3
"""CPU-only guards for the response races exposed by overlap migration."""

import asyncio
import inspect
import unittest

from sglang.multi_model.request_handler_worker_pool import (
    ReqState,
    RequestHandlerWorkerPool,
)


class V6ResponseDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_queue_preserves_back_to_back_final_edge(self):
        state = ReqState([], False, asyncio.Event(), asyncio.Queue())
        state.response_queue.put_nowait(({"text": "partial"}, False))
        state.response_queue.put_nowait(({"text": "final"}, True))

        self.assertEqual(
            await state.response_queue.get(), ({"text": "partial"}, False)
        )
        self.assertEqual(
            await state.response_queue.get(), ({"text": "final"}, True)
        )

    async def test_request_is_registered_before_redis_publish(self):
        source = inspect.getsource(RequestHandlerWorkerPool._send_single_request)
        registered = source.index("self.rid_to_state[single_request_obj.rid] = state")
        published = source.index("await self.redis_client.send_pyobj")
        self.assertLess(registered, published)

    async def test_generation_waiter_does_not_clear_shared_event(self):
        source = inspect.getsource(RequestHandlerWorkerPool._wait_for_response)
        self.assertIn("state.response_queue.get()", source)
        self.assertNotIn("state.event.clear()", source)


if __name__ == "__main__":
    unittest.main()
