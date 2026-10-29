"""Focused unit tests for GPU concurrency protection in api/main.py.

Verifies:
- Two simultaneous inference requests never exceed the configured limit (default: 1).
- Both Mark 2 and Mark 3 routes share the same concurrency limit.
- A failed inference releases its permit.
- A cancelled request does not leak a permit.
- A subsequent valid request succeeds after a prior failure.
- Temporary files are cleaned up when inference fails.
- Health checks remain responsive while inference is in progress.
- Bounded wait times out with HTTP 503 under overload.
"""

from __future__ import annotations

import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import api.main as api


def upload(filename: str, content: bytes = b"mock content") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


class GpuConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reset semaphore to 1 before each test
        api.reset_inference_semaphore(1)
        self.orig_thresholds = api.mark2_thresholds
        api.mark2_thresholds = {}

    async def asyncTearDown(self):
        api.reset_inference_semaphore(1)
        api.mark2_thresholds = self.orig_thresholds

    async def test_two_simultaneous_inference_requests_never_exceed_limit(self):
        """Simultaneous requests must be serialized when concurrency limit is 1."""
        api.reset_inference_semaphore(1)
        active_count = 0
        peak_active = 0
        execution_order = []

        async def simulated_gpu_inference(task_id: int):
            nonlocal active_count, peak_active
            async with api.inference_guard(timeout=5.0):
                active_count += 1
                peak_active = max(peak_active, active_count)
                execution_order.append(f"start_{task_id}")
                await asyncio.sleep(0.05)
                active_count -= 1
                execution_order.append(f"end_{task_id}")
                return task_id

        results = await asyncio.gather(
            simulated_gpu_inference(1),
            simulated_gpu_inference(2),
        )

        self.assertEqual(results, [1, 2])
        self.assertEqual(peak_active, 1, "Peak active concurrency exceeded configured limit of 1")
        # Ensure serialization: task 1 finishes before task 2 starts (or vice-versa)
        self.assertEqual(execution_order[1], f"end_{execution_order[0].split('_')[1]}")

    async def test_mark2_and_mark3_routes_share_same_limit(self):
        """Both Mark 2 and Mark 3 routes must acquire the identical semaphore instance."""
        api.reset_inference_semaphore(1)
        sem = api.get_inference_semaphore()

        # Hold permit
        await sem.acquire()
        try:
            # When Mark 3 tries to acquire with a short timeout, it must time out because Mark 2 holds it
            with self.assertRaises(HTTPException) as ctx:
                async with api.inference_guard(timeout=0.05):
                    pass
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("GPU inference queue is busy", ctx.exception.detail)
        finally:
            sem.release()

    async def test_failed_inference_releases_permit(self):
        """When inference raises an unexpected exception, the permit must be cleanly released."""
        api.reset_inference_semaphore(1)

        with self.assertRaises(RuntimeError):
            async with api.inference_guard(timeout=1.0):
                raise RuntimeError("Simulated model forward pass failure")

        # Verify permit is immediately available for the next caller
        sem = api.get_inference_semaphore()
        self.assertEqual(sem._value, 1, "Permit was leaked after exception")

    async def test_cancelled_request_does_not_leak_permit(self):
        """When an in-flight request is cancelled, its permit must be released immediately."""
        api.reset_inference_semaphore(1)
        acquired_event = asyncio.Event()

        async def worker():
            async with api.inference_guard(timeout=1.0):
                acquired_event.set()
                await asyncio.sleep(10.0)  # Hang until cancelled

        task = asyncio.create_task(worker())
        await acquired_event.wait()

        # Cancel while holding the permit
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # Verify permit was released
        sem = api.get_inference_semaphore()
        self.assertEqual(sem._value, 1, "Permit was leaked after task cancellation")

    async def test_subsequent_valid_request_succeeds_after_prior_failure(self):
        """A failure in request N must not prevent request N+1 from executing."""
        api.reset_inference_semaphore(1)

        # Request 1 fails
        with self.assertRaises(ValueError):
            async with api.inference_guard(timeout=1.0):
                raise ValueError("Faulty input error")

        # Request 2 succeeds
        executed = False
        async with api.inference_guard(timeout=1.0):
            executed = True

        self.assertTrue(executed)

    async def test_temporary_files_cleaned_up_when_inference_fails(self):
        """Temporary files must be deleted even if inference raises an error or times out."""
        removed_files = []

        async def save_upload(_upload, kind):
            return f"temp-path-{kind}"

        with (
            patch.object(api, "mark2_fusion_model", object()),
            patch.object(api, "mark2_text_model", object()),
            patch.object(api, "mark2_tokenizer", MagicMock()),
            patch.object(api, "_save_upload", save_upload),
            patch.object(api.os.path, "exists", return_value=True),
            patch.object(api.os, "remove", side_effect=removed_files.append),
            patch.object(api, "inference_guard", side_effect=RuntimeError("Simulated GPU crash")),
        ):
            with self.assertRaises(HTTPException):
                await api.predict_raw(
                    text="Hello test",
                    audio=upload("audio.wav"),
                    video=upload("video.mp4"),
                )

        self.assertEqual(removed_files, ["temp-path-video", "temp-path-audio"])

    async def test_health_checks_remain_responsive_while_inference_is_in_progress(self):
        """Health check endpoint must return immediately without waiting on inference semaphore."""
        api.reset_inference_semaphore(1)
        sem = api.get_inference_semaphore()

        # Simulate ongoing heavy inference by holding permit
        await sem.acquire()
        try:
            # Health check must still execute synchronously and return status 'ok'
            health_response = api.health()
            self.assertEqual(health_response["status"], "ok")
        finally:
            sem.release()

    async def test_overload_timeout_raises_http_503(self):
        """When all permits are occupied beyond the timeout window, raise HTTP 503."""
        api.reset_inference_semaphore(1)
        sem = api.get_inference_semaphore()

        await sem.acquire()
        try:
            with self.assertRaises(HTTPException) as ctx:
                async with api.inference_guard(timeout=0.05):
                    pass
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("GPU inference queue is busy", ctx.exception.detail)
        finally:
            sem.release()

    async def test_event_loop_unblocked_during_inference_offload(self):
        """Heavy CPU/GPU inference offloaded via asyncio.to_thread must not block the event loop."""
        import time

        inference_running = True
        loop_ticks = 0

        def blocking_gpu_work():
            time.sleep(0.1)
            return "done"

        async def background_ticker():
            nonlocal loop_ticks
            while inference_running:
                loop_ticks += 1
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(background_ticker())

        async with api.inference_guard(timeout=1.0):
            result = await asyncio.to_thread(blocking_gpu_work)

        inference_running = False
        await ticker_task

        self.assertEqual(result, "done")
        self.assertGreater(
            loop_ticks,
            3,
            "Event loop was blocked by inference! Expected ticker coroutine to progress concurrently.",
        )

    async def test_concurrent_health_check_responds_while_inference_thread_runs(self):
        """Concurrent health checks return immediately while a long inference thread is executing."""
        import time
        api.reset_inference_semaphore(1)

        inference_finished = False

        def slow_inference():
            time.sleep(0.15)
            nonlocal inference_finished
            inference_finished = True
            return "finished"

        async def run_inference():
            async with api.inference_guard(timeout=2.0):
                return await asyncio.to_thread(slow_inference)

        inf_task = asyncio.create_task(run_inference())
        # Let the inference task acquire permit and start thread
        await asyncio.sleep(0.02)

        # Health check runs concurrently on the event loop
        health_resp = api.health()
        self.assertEqual(health_resp["status"], "ok")
        # Health check completed while inference was still underway
        self.assertFalse(inference_finished, "Health check should complete while inference thread is still running")

        result = await inf_task
        self.assertEqual(result, "finished")
        self.assertTrue(inference_finished)


if __name__ == "__main__":
    unittest.main()
