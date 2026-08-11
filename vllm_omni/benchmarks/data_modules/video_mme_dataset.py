"""Video-MME dataset adapter for ``vllm bench serve``.

Annotations can be loaded from ``lmms-lab/Video-MME`` or from an official
Video-MME result-template JSON. Benchmark videos are intentionally local-only:
the official dataset license does not permit vLLM-Omni to redistribute them.
"""

from __future__ import annotations

import base64
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
except ImportError:
    from vllm.benchmarks.datasets import HuggingFaceDataset as BenchmarkDataset
    from vllm.benchmarks.datasets import SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

logger = logging.getLogger(__name__)

VIDEO_MME_OFFICIAL_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)


@dataclass
class VideoMMESampleRequest(SampleRequest):
    video_mme_gold_answer: str = ""
    video_mme_video_id: str = ""
    video_mme_question_id: str = ""
    video_mme_duration: str = ""
    video_mme_domain: str = ""
    video_mme_sub_category: str = ""
    video_mme_task_type: str = ""
    omni_extra_body: dict[str, Any] | None = None
    omni_chat_messages: list[dict[str, Any]] | None = None


class _ListDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


class VideoMMEDataset(BenchmarkDataset):
    """Official Video-MME multiple-choice questions with local video media."""

    SUPPORTED_DATASET_PATHS = {"lmms-lab/Video-MME"}
    DEFAULT_HF_DATASET_ID = "lmms-lab/Video-MME"
    DEFAULT_OUTPUT_LEN = 16
    IS_MULTIMODAL = True

    def __init__(
        self,
        *,
        annotations_json: str | None = None,
        dataset_path: str | None = None,
        dataset_split: str = "test",
        dataset_subset: str | None = "videomme",
        video_dir: str,
        duration: str = "all",
        inline_local_video: bool = False,
        random_seed: int = 0,
        no_stream: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> None:
        if duration not in ("all", "short", "medium", "long"):
            raise ValueError(f"Video-MME duration must be all/short/medium/long, got {duration!r}")
        if not video_dir:
            raise ValueError("Video-MME requires --video-mme-video-dir; benchmark media is not redistributed")

        self.annotations_json = Path(annotations_json).expanduser() if annotations_json else None
        self.dataset_path = dataset_path or self.DEFAULT_HF_DATASET_ID
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.video_dir = Path(video_dir).expanduser().resolve()
        self.duration = duration
        self.inline_local_video = inline_local_video
        self._hf_streaming = not no_stream
        self.trust_remote_code = trust_remote_code

        if not self.video_dir.is_dir():
            raise FileNotFoundError(f"Video-MME video directory not found: {self.video_dir}")
        super().__init__(dataset_path=None, random_seed=random_seed, **kwargs)
        self.load_data()

    @staticmethod
    def _flatten_official_json(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            raise ValueError("Video-MME annotations JSON must contain a list")
        rows: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            questions = item.get("questions")
            if not isinstance(questions, list):
                rows.append(dict(item))
                continue
            shared = {k: v for k, v in item.items() if k != "questions"}
            for question in questions:
                if isinstance(question, dict):
                    rows.append({**shared, **question})
        return rows

    def load_data(self) -> None:
        if self.annotations_json is not None:
            if not self.annotations_json.is_file():
                raise FileNotFoundError(f"Video-MME annotations JSON not found: {self.annotations_json}")
            rows = self._flatten_official_json(json.loads(self.annotations_json.read_text(encoding="utf-8")))
            if not getattr(self, "disable_shuffle", False):
                random.Random(self.random_seed).shuffle(rows)
            self.data = _ListDataset(rows)
            return

        if load_dataset is None:
            raise ImportError("Video-MME Hub annotations require the 'datasets' package")
        kwargs: dict[str, Any] = {
            "path": self.dataset_path,
            "split": self.dataset_split,
            "streaming": self._hf_streaming,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dataset_subset:
            kwargs["name"] = self.dataset_subset
        data = load_dataset(**kwargs)
        if not getattr(self, "disable_shuffle", False):
            data = data.shuffle(seed=self.random_seed)
        self.data = data

    def _resolve_video(self, row: dict[str, Any]) -> Path | None:
        video_id = str(row.get("videoID") or row.get("video_id") or "").strip()
        short_id = str(row.get("video_id") or "").strip()
        candidates = [
            self.video_dir / f"{video_id}.mp4",
            self.video_dir / video_id,
            self.video_dir / f"{short_id}.mp4",
            self.video_dir / short_id / f"{short_id}.mp4",
        ]
        for path in candidates:
            if path.is_file():
                return path.resolve()
        logger.warning("Skipping Video-MME row: no local video for videoID=%r video_id=%r", video_id, short_id)
        return None

    def _video_content(self, path: Path) -> dict[str, Any]:
        if self.inline_local_video:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            url = f"data:video/mp4;base64,{encoded}"
        else:
            url = path.as_uri()
        return {"type": "video_url", "video_url": {"url": url}}

    @staticmethod
    def _user_prompt(row: dict[str, Any]) -> str:
        question = str(row.get("question") or "").strip()
        options = row.get("options") or []
        option_text = "\n".join(str(option).strip() for option in options if str(option).strip())
        return f"{VIDEO_MME_OFFICIAL_PROMPT}\n{question}\n{option_text}\nThe best answer is:"

    def _make_request(
        self,
        row: dict[str, Any],
        tokenizer: TokenizerLike,
        output_len: int,
        request_id: str,
    ) -> VideoMMESampleRequest | None:
        duration = str(row.get("duration") or "").strip().lower()
        if self.duration != "all" and duration != self.duration:
            return None
        video_path = self._resolve_video(row)
        if video_path is None:
            return None
        prompt = self._user_prompt(row)
        messages = [
            {
                "role": "user",
                "content": [self._video_content(video_path), {"type": "text", "text": prompt}],
            }
        ]
        video_id = str(row.get("videoID") or row.get("video_id") or "").strip()
        return VideoMMESampleRequest(
            prompt=prompt,
            prompt_len=len(tokenizer.encode(prompt)),
            expected_output_len=output_len,
            multi_modal_data=None,
            request_id=request_id,
            video_mme_gold_answer=str(row.get("answer") or "").strip(),
            video_mme_video_id=video_id,
            video_mme_question_id=str(row.get("question_id") or "").strip(),
            video_mme_duration=duration,
            video_mme_domain=str(row.get("domain") or "").strip(),
            video_mme_sub_category=str(row.get("sub_category") or "").strip(),
            video_mme_task_type=str(row.get("task_type") or "").strip(),
            omni_extra_body={"modalities": ["text"]},
            omni_chat_messages=messages,
        )

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        output_len = output_len or self.DEFAULT_OUTPUT_LEN
        tokenizer = get_cached_tokenizer(tokenizer)
        requests: list[SampleRequest] = []
        for row in self.data:
            if len(requests) >= num_requests:
                break
            if not isinstance(row, dict):
                row = dict(row)
            request = self._make_request(row, tokenizer, output_len, f"{request_id_prefix}{len(requests)}")
            if request is not None:
                requests.append(request)
        self.maybe_oversample_requests(requests, num_requests, request_id_prefix, no_oversample)
        return requests
