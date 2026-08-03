"""Video-MME multiple-choice accuracy scoring."""

from __future__ import annotations

from typing import Any

from vllm_omni.benchmarks.data_modules.daily_omni_eval import extract_predicted_choice
from vllm_omni.benchmarks.data_modules.video_mme_dataset import VideoMMESampleRequest


def compute_video_mme_accuracy_metrics(
    input_requests: list[Any], outputs: list[Any], *, include_per_item: bool = False
) -> dict[str, Any] | None:
    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(request, VideoMMESampleRequest) for request in input_requests):
        return None

    correct = 0
    evaluated = 0
    evaluated_ok = 0
    request_failed = 0
    parse_failed = 0
    groups: dict[str, dict[str, dict[str, int]]] = {
        "duration": {},
        "domain": {},
        "sub_category": {},
        "task_type": {},
    }
    items: list[dict[str, Any]] = []

    def update_group(kind: str, name: str, *, is_correct: bool, success: bool) -> None:
        key = name or "unknown"
        stats = groups[kind].setdefault(key, {"correct": 0, "total": 0, "total_ok": 0})
        stats["total"] += 1
        if success:
            stats["total_ok"] += 1
        if is_correct:
            stats["correct"] += 1

    for request, output in zip(input_requests, outputs, strict=True):
        gold = request.video_mme_gold_answer.strip().upper()
        if not gold:
            continue
        evaluated += 1
        success = bool(getattr(output, "success", False))
        if success:
            evaluated_ok += 1
            predicted = extract_predicted_choice(getattr(output, "generated_text", ""))
            if predicted is None:
                parse_failed += 1
        else:
            request_failed += 1
            predicted = None
        is_correct = bool(success and predicted == gold)
        correct += int(is_correct)
        dimensions = {
            "duration": request.video_mme_duration,
            "domain": request.video_mme_domain,
            "sub_category": request.video_mme_sub_category,
            "task_type": request.video_mme_task_type,
        }
        for kind, name in dimensions.items():
            update_group(kind, name, is_correct=is_correct, success=success)
        if include_per_item:
            items.append(
                {
                    "request_id": request.request_id,
                    "video_id": request.video_mme_video_id,
                    "question_id": request.video_mme_question_id,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": is_correct,
                    **dimensions,
                }
            )

    result: dict[str, Any] = {
        "video_mme_accuracy": (correct / evaluated_ok) if evaluated_ok else None,
        "video_mme_accuracy_incl_http_fail": (correct / evaluated) if evaluated else None,
        "video_mme_correct": correct,
        "video_mme_evaluated": evaluated,
        "video_mme_evaluated_ok": evaluated_ok,
        "video_mme_request_failed": request_failed,
        "video_mme_parse_failed": parse_failed,
    }
    for kind, values in groups.items():
        result[f"video_mme_per_{kind}"] = values
        result[f"video_mme_per_{kind}_accuracy"] = {
            name: (stats["correct"] / stats["total_ok"]) if stats["total_ok"] else None
            for name, stats in values.items()
        }
    if include_per_item:
        result["video_mme_eval_items"] = items
    return result


def print_video_mme_accuracy_summary(metrics: dict[str, Any]) -> None:
    accuracy = metrics.get("video_mme_accuracy")
    if accuracy is None:
        return
    correct = int(metrics.get("video_mme_correct", 0) or 0)
    total = int(metrics.get("video_mme_evaluated_ok", 0) or 0)
    print("{s:{c}^{n}}".format(s=" Video-MME accuracy (MCQ) ", n=50, c="="))
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.2%}")
