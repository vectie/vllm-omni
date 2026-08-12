# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ComparableRequest(SimpleNamespace):
    def __lt__(self, other) -> bool:
        return (self.priority, self.arrival_time, self.request_id) < (
            other.priority,
            other.arrival_time,
            other.request_id,
        )


def test_background_aging_promotes_waiting_request(monkeypatch) -> None:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.running = []
    foreground = _ComparableRequest(request_id="foreground", priority=-100, arrival_time=99.0)
    background = _ComparableRequest(request_id="background", priority=0, arrival_time=90.0)
    scheduler.waiting.add_request(foreground)
    scheduler.waiting.add_request(background)
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_BACKGROUND_AGING_S", "2")
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time", lambda: 100.0)

    scheduler._promote_aged_background_requests()

    assert scheduler.waiting.peek_request() is background
    assert background.priority == -101


def test_background_aging_leaves_recent_request_behind_foreground(monkeypatch) -> None:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.running = []
    foreground = _ComparableRequest(request_id="foreground", priority=-100, arrival_time=99.0)
    background = _ComparableRequest(request_id="background", priority=0, arrival_time=99.5)
    scheduler.waiting.add_request(foreground)
    scheduler.waiting.add_request(background)
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_BACKGROUND_AGING_S", "2")
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time", lambda: 100.0)

    scheduler._promote_aged_background_requests()

    assert scheduler.waiting.peek_request() is foreground
    assert background.priority == 0


def test_remove_queued_request_supports_priority_queue_and_running_list() -> None:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    waiting = create_request_queue(SchedulingPolicy.PRIORITY)
    request = _ComparableRequest(request_id="aborted", priority=-100, arrival_time=99.0)
    waiting.add_request(request)
    running = [request]

    scheduler._remove_queued_request(waiting, request)
    scheduler._remove_queued_request(running, request)

    assert not waiting
    assert running == []
