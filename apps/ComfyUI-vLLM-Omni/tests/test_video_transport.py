import asyncio
import unittest

from aiohttp import web
from comfyui_vllm_omni.utils.video_transport import run_video_job


class VideoTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode = "complete"
        self.posts = 0
        self.deletes = 0
        self.key = None
        self.submitted = asyncio.Event()
        app = web.Application()
        app.router.add_route("*", "/v1/videos", self.handle)
        app.router.add_route("*", "/v1/videos/{tail:.*}", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def handle(self, request):
        if request.method == "POST":
            await request.read()
            self.posts += 1
            self.key = request.headers.get("Idempotency-Key")
            self.submitted.set()
            return web.json_response(
                {"id": "../escape" if self.mode == "bad-id" else "video-fixture", "status": "queued"}
            )
        if request.method == "DELETE":
            self.deletes += 1
            return web.json_response({"id": "video-fixture", "deleted": True})
        if request.path.endswith("/content"):
            return web.Response(body=b"fixture-not-real-video", content_type="video/mp4")
        if self.mode == "failed":
            return web.json_response(
                {"id": "video-fixture", "status": "failed", "error": "PRIVATE PROMPT AND TRACE"}, status=500
            )
        return web.json_response({"id": "video-fixture", "status": "queued" if self.mode == "pending" else "completed"})

    async def run_job(self, **kwargs):
        return await run_video_job(self.base, {"model": "fixture", "prompt": "private"}, poll_interval=0.01, **kwargs)

    async def test_success_has_idempotency_and_cleanup(self):
        self.assertEqual(await self.run_job(), b"fixture-not-real-video")
        self.assertRegex(self.key, r"^[a-f0-9]{32}$")
        self.assertEqual((self.posts, self.deletes), (1, 1))

    async def test_deadline_cleans_up_without_reposting(self):
        self.mode = "pending"
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            await self.run_job(max_poll_duration=0.05)
        self.assertEqual((self.posts, self.deletes), (1, 1))

    async def test_task_cancellation_cleans_up(self):
        self.mode = "pending"
        task = asyncio.create_task(self.run_job())
        await self.submitted.wait()
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((self.posts, self.deletes), (1, 1))

    async def test_failure_is_redacted_and_cleaned_up(self):
        self.mode = "failed"
        with self.assertRaises(RuntimeError) as raised:
            await self.run_job()
        self.assertNotIn("PRIVATE", str(raised.exception))
        self.assertEqual(self.deletes, 1)

    async def test_oversize_download_is_cleaned_up(self):
        with self.assertRaisesRegex(RuntimeError, "download limit"):
            await self.run_job(maximum_result_bytes=1)
        self.assertEqual(self.deletes, 1)

    async def test_untrusted_job_id_never_becomes_a_url(self):
        self.mode = "bad-id"
        with self.assertRaisesRegex(RuntimeError, "identifier"):
            await self.run_job()
        self.assertEqual((self.posts, self.deletes), (1, 0))
