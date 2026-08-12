# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

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


def _preemption_scheduler(*requests: _ComparableRequest) -> OmniARScheduler:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.running = list(requests)
    scheduler.max_num_running_reqs = 2
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._omni_aged_background = None
    return scheduler


def _request(request_id: str, *, priority: int, arrival_time: float, status: RequestStatus) -> _ComparableRequest:
    return _ComparableRequest(
        request_id=request_id,
        priority=priority,
        arrival_time=arrival_time,
        status=status,
        is_finished=lambda: False,
    )


def test_foreground_preemption_reclaims_one_full_stage_slot(monkeypatch) -> None:
    older = _request("older", priority=0, arrival_time=1.0, status=RequestStatus.RUNNING)
    newer = _request("newer", priority=0, arrival_time=2.0, status=RequestStatus.RUNNING)
    foreground = _request("duplex", priority=-100, arrival_time=3.0, status=RequestStatus.WAITING)
    scheduler = _preemption_scheduler(older, newer)
    scheduler.waiting.add_request(foreground)
    preempted: list[_ComparableRequest] = []

    def _preempt(request, _timestamp) -> None:
        preempted.append(request)

    scheduler._preempt_request = _preempt
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_FOREGROUND_PREEMPTION_PRIORITY", "-100")

    victim = scheduler._preempt_background_for_foreground()

    assert victim is newer
    assert preempted == [newer]
    assert scheduler.running == [older]


def test_foreground_preemption_is_work_conserving_when_slot_is_free(monkeypatch) -> None:
    background = _request("background", priority=0, arrival_time=1.0, status=RequestStatus.RUNNING)
    foreground = _request("duplex", priority=-100, arrival_time=2.0, status=RequestStatus.WAITING)
    scheduler = _preemption_scheduler(background)
    scheduler.waiting.add_request(foreground)
    scheduler._preempt_request = lambda *_args: pytest.fail("preemption should not run with a free slot")
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_FOREGROUND_PREEMPTION_PRIORITY", "-100")

    assert scheduler._preempt_background_for_foreground() is None


def test_aged_background_is_not_misclassified_as_foreground(monkeypatch) -> None:
    running = _request("running", priority=0, arrival_time=1.0, status=RequestStatus.RUNNING)
    second = _request("second", priority=0, arrival_time=2.0, status=RequestStatus.RUNNING)
    aged = _request("aged", priority=-101, arrival_time=0.0, status=RequestStatus.WAITING)
    foreground = _request("duplex", priority=-100, arrival_time=3.0, status=RequestStatus.WAITING)
    scheduler = _preemption_scheduler(running, second)
    scheduler.waiting.add_request(aged)
    scheduler.waiting.add_request(foreground)
    scheduler._omni_aged_background = (aged, 0)
    scheduler._preempt_request = lambda *_args: pytest.fail("aged background must not trigger foreground preemption")
    monkeypatch.setenv("VLLM_OMNI_DUPLEX_FOREGROUND_PREEMPTION_PRIORITY", "-100")

    assert scheduler._preempt_background_for_foreground() is None
