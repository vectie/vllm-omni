"""Bounded asynchronous video transport; no model execution or credentials here."""

import asyncio
import json
import logging
import re
import uuid

import aiohttp

logger = logging.getLogger(__name__)
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


async def _json(session, verb, url, **kwargs):
    async with session.request(verb, url, allow_redirects=False, **kwargs) as response:
        if 300 <= response.status < 400:
            raise RuntimeError("Video service redirects are not permitted")
        chunks = bytearray()
        async for chunk in response.content.iter_chunked(8192):
            chunks.extend(chunk)
            if len(chunks) > 262144:
                raise RuntimeError("Video service metadata exceeds its limit")
        try:
            data = json.loads(chunks)
        except (ValueError, UnicodeError):
            raise RuntimeError("Video service returned invalid metadata") from None
        if not isinstance(data, dict):
            raise RuntimeError("Video service returned invalid metadata")
        if response.status == 409 and verb == "DELETE":
            return {"deleted": False}
        # vLLM-Omni can return a failed job object with an error HTTP status.
        if not 200 <= response.status < 300 and not (verb == "GET" and data.get("status") == "failed"):
            raise RuntimeError(f"Video service rejected the operation (HTTP {response.status})")
        return data


async def _cleanup(session, url, job_id):
    for _ in range(4):
        result = await _json(session, "DELETE", url)
        if result.get("id") == job_id and result.get("deleted") is True:
            return
        await asyncio.sleep(1)
    raise RuntimeError("Video cancellation has not been acknowledged")


async def run_video_job(
    base_url,
    form,
    *,
    timeout=None,
    poll_interval=5.0,
    max_poll_duration=7200.0,
    maximum_result_bytes=1024 * 1024 * 1024,
):
    """POST exactly once, poll within a deadline, and clean up every known job.

    The random idempotency key identifies this enqueue attempt; it is not an
    authentication secret. An ambiguous POST is never automatically retried.
    Server-side reconciliation remains necessary if cancellation cannot reach
    the service or no job identifier was received.
    """
    if not 0 < poll_interval <= 60 or not 0 < max_poll_duration <= 7200:
        raise ValueError("Video polling limits are invalid")
    if not 0 < maximum_result_bytes <= 1024 * 1024 * 1024:
        raise ValueError("Video result limit is invalid")
    timeout = timeout or aiohttp.ClientTimeout(total=60)
    job_id = None
    url = base_url.rstrip("/") + "/videos"
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with asyncio.timeout(max_poll_duration):
                data = await _json(session, "POST", url, data=form, headers={"Idempotency-Key": uuid.uuid4().hex})
                candidate = data.get("id")
                if not isinstance(candidate, str) or not _JOB_ID.fullmatch(candidate):
                    raise RuntimeError("Video service returned an invalid job identifier")
                job_id = candidate
                url += "/" + job_id
                while True:
                    if data.get("id") != job_id:
                        raise RuntimeError("Video service returned a different job")
                    status = data.get("status")
                    if status == "completed":
                        break
                    if status in {"failed", "cancelled", "expired"}:
                        raise RuntimeError("Video generation failed or its access ended")
                    if status not in {"queued", "in_progress"}:
                        raise RuntimeError("Video service returned an invalid job status")
                    await asyncio.sleep(poll_interval)
                    data = await _json(session, "GET", url)
                async with session.get(url + "/content", allow_redirects=False) as response:
                    if response.status != 200 or response.content_type != "video/mp4":
                        raise RuntimeError("Completed video content is unavailable")
                    result = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        result.extend(chunk)
                        if len(result) > maximum_result_bytes:
                            raise RuntimeError("Video result exceeds its download limit")
                    if not result:
                        raise RuntimeError("Video service returned empty content")
                    return bytes(result)
        except (aiohttp.ClientError, TimeoutError):
            raise RuntimeError(
                "Video operation timed out or disconnected; do not enqueue another copy until its status is reconciled"
            ) from None
        finally:
            if job_id is not None:
                try:
                    await asyncio.wait_for(_cleanup(session, url, job_id), timeout=10)
                except Exception:
                    # Do not log provider error bodies, URLs, prompts or tokens.
                    logger.warning("Video cleanup was not confirmed; server reconciliation is required")
