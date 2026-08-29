#!/usr/bin/env python3
"""Screen a graph-captured MiniCPM-o 4.5 codec sampler on Ascend."""

from __future__ import annotations

import statistics
import time
import os

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


HIDDEN = 768
VOCAB = 6562
TOP_K = 25
TOP_P = 0.85
TEMPERATURE = 0.8
PENALTY = 1.05


def sample_step(
    hidden: torch.Tensor,
    frequencies: torch.Tensor,
    weight: torch.Tensor,
    uniform: torch.Tensor,
    expired: torch.Tensor,
    allow_eos: torch.Tensor,
    penalty: torch.Tensor,
    vocab_ids: torch.Tensor,
    frequency_output: torch.Tensor,
    *,
    exponential_race: bool = False,
) -> torch.Tensor:
    logits = F.linear(hidden, weight).float() / TEMPERATURE
    alpha = torch.pow(penalty, frequencies)
    logits = torch.where(logits < 0, logits * alpha, logits / alpha)
    eos = VOCAB - 1
    eos_value = torch.where(
        allow_eos,
        logits[..., eos],
        logits.new_full(logits[..., eos].shape, float("-inf")),
    )
    logits[..., eos] = eos_value

    candidates, candidate_ids = torch.topk(logits, TOP_K, dim=-1)
    max_logits = logits.amax(dim=-1, keepdim=True)
    total_mass = torch.exp(logits - max_logits).sum(dim=-1, keepdim=True)
    candidate_mass = torch.exp(candidates - max_logits)
    outside_mass = (total_mass - candidate_mass.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    candidates = candidates.flip(-1)
    candidate_ids = candidate_ids.flip(-1)
    candidate_mass = candidate_mass.flip(-1)
    cumulative = (outside_mass + candidate_mass.cumsum(dim=-1)) / total_mass
    remove = cumulative <= (1.0 - TOP_P)
    remove[..., -3:] = False
    candidates = candidates.masked_fill(remove, float("-inf"))
    probabilities = torch.softmax(candidates, dim=-1)
    if exponential_race:
        sampled_position = (probabilities / uniform).argmax(dim=-1, keepdim=True)
    else:
        sampled_position = torch.sum(probabilities.cumsum(dim=-1) < uniform, dim=-1, keepdim=True)
        sampled_position = sampled_position.clamp_max_(TOP_K - 1)
    sampled = candidate_ids.gather(-1, sampled_position)

    next_frequencies = frequencies + (vocab_ids == sampled).to(frequencies.dtype)
    next_frequencies = next_frequencies - (
        (expired >= 0) & (vocab_ids == expired)
    ).to(frequencies.dtype)
    frequency_output.copy_(next_frequencies)
    return sampled


def filter_step(
    hidden: torch.Tensor,
    frequencies: torch.Tensor,
    weight: torch.Tensor,
    allow_eos: torch.Tensor,
    penalty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden, weight).float() / TEMPERATURE
    alpha = torch.pow(penalty, frequencies)
    logits = torch.where(logits < 0, logits * alpha, logits / alpha)
    eos = VOCAB - 1
    eos_value = torch.where(
        allow_eos,
        logits[..., eos],
        logits.new_full(logits[..., eos].shape, float("-inf")),
    )
    logits[..., eos] = eos_value
    candidates, candidate_ids = torch.topk(logits, TOP_K, dim=-1)
    max_logits = logits.amax(dim=-1, keepdim=True)
    total_mass = torch.exp(logits - max_logits).sum(dim=-1, keepdim=True)
    candidate_mass = torch.exp(candidates - max_logits)
    outside_mass = (total_mass - candidate_mass.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    candidates = candidates.flip(-1)
    candidate_ids = candidate_ids.flip(-1)
    candidate_mass = candidate_mass.flip(-1)
    cumulative = (outside_mass + candidate_mass.cumsum(dim=-1)) / total_mass
    remove = cumulative <= (1.0 - TOP_P)
    remove[..., -3:] = False
    candidates = candidates.masked_fill(remove, float("-inf"))
    return torch.softmax(candidates, dim=-1), candidate_ids


def advance_frequency(
    frequencies: torch.Tensor,
    sampled: torch.Tensor,
    expired: torch.Tensor,
    vocab_ids: torch.Tensor,
) -> torch.Tensor:
    result = frequencies + (vocab_ids == sampled).to(frequencies.dtype)
    return result - ((expired >= 0) & (vocab_ids == expired)).to(frequencies.dtype)


def event_us(fn, iterations: int = 200) -> list[float]:
    values = []
    for _ in range(iterations):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def main() -> None:
    allow_internal_format = os.environ.get("BENCH_ALLOW_INTERNAL_FORMAT", "0") == "1"
    exact_only = os.environ.get("BENCH_EXACT_ONLY", "0") == "1"
    nd_boundary = os.environ.get("BENCH_ND_BOUNDARY", "0") == "1"
    two_graphs = os.environ.get("BENCH_TWO_GRAPHS", "0") == "1"
    exponential_race = os.environ.get("BENCH_EXPONENTIAL_RACE", "0") == "1"
    torch.npu.config.allow_internal_format = allow_internal_format
    print(
        f"allow_internal_format={allow_internal_format} "
        f"exact_only={exact_only} nd_boundary={nd_boundary} two_graphs={two_graphs} "
        f"exponential_race={exponential_race}"
    )
    torch.manual_seed(7)
    device = torch.device(os.environ.get("BENCH_DEVICE", "npu"))
    hidden = torch.randn((1, HIDDEN), device=device, dtype=torch.bfloat16)
    weight = torch.randn((VOCAB, HIDDEN), device=device, dtype=torch.bfloat16) * 0.02
    freq0 = torch.zeros((1, VOCAB), device=device, dtype=torch.float32)
    freq1 = torch.zeros_like(freq0)
    uniform = torch.full(
        (1, TOP_K if exponential_race else 1),
        0.37,
        device=device,
        dtype=torch.float32,
    )
    expired = torch.full((1, 1), -1, device=device, dtype=torch.long)
    allow_eos = torch.zeros((1,), device=device, dtype=torch.bool)
    penalty = torch.full((1,), PENALTY, device=device, dtype=torch.float32)
    vocab_ids = torch.arange(VOCAB, device=device, dtype=torch.long)

    if not exact_only:
        probabilities = torch.softmax(torch.randn((1, TOP_K), device=device), dim=-1)
        parity = 0
        state_parity = 0
        for seed in range(1000):
            multinomial_generator = torch.Generator(device=device).manual_seed(seed)
            exponential_generator = torch.Generator(device=device).manual_seed(seed)
            expected = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=multinomial_generator,
            )
            noise = torch.empty_like(probabilities).exponential_(
                1.0,
                generator=exponential_generator,
            )
            actual = torch.argmax(probabilities / noise, dim=-1, keepdim=True)
            parity += int(torch.equal(expected, actual))
            state_parity += int(
                torch.equal(
                    multinomial_generator.get_state(),
                    exponential_generator.get_state(),
                )
            )
        print(f"multinomial_exponential_seed_parity={parity}/1000")
        print(f"multinomial_exponential_rng_state_parity={state_parity}/1000")

        for _ in range(5):
            sample_step(
                hidden,
                freq0,
                weight,
                uniform,
                expired,
                allow_eos,
                penalty,
                vocab_ids,
                freq1,
                exponential_race=exponential_race,
            )
        torch.npu.synchronize()
        eager = event_us(
            lambda: sample_step(
                hidden,
                freq0,
                weight,
                uniform,
                expired,
                allow_eos,
                penalty,
                vocab_ids,
                freq1,
                exponential_race=exponential_race,
            )
        )
        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.inference_mode(), torch.npu.graph(graph, pool=pool):
            sampled = sample_step(
                hidden,
                freq0,
                weight,
                uniform,
                expired,
                allow_eos,
                penalty,
                vocab_ids,
                freq1,
                exponential_race=exponential_race,
            )
        graph.replay()
        torch.npu.synchronize()
        replay = event_us(graph.replay)

        graph_rng = torch.Generator(device=device).manual_seed(42)

        def stochastic_graph_step() -> None:
            if exponential_race:
                uniform.exponential_(1.0, generator=graph_rng)
            else:
                uniform.uniform_(0.0, 1.0, generator=graph_rng)
            graph.replay()

        stochastic_replay = event_us(stochastic_graph_step)

        before = freq1.clone()
        uniform.fill_(0.91)
        graph.replay()
        torch.npu.synchronize()
        changed = not torch.equal(before, freq1)
        print(f"sample={int(sampled.item())} runtime_input_changed_output={changed}")
        print(
            "eager_us "
            f"mean={statistics.mean(eager):.3f} median={statistics.median(eager):.3f} "
            f"p99={sorted(eager)[int(len(eager) * 0.99) - 1]:.3f}"
        )
        print(
            "graph_us "
            f"mean={statistics.mean(replay):.3f} median={statistics.median(replay):.3f} "
            f"p99={sorted(replay)[int(len(replay) * 0.99) - 1]:.3f}"
        )
        print(f"speedup={statistics.mean(eager) / statistics.mean(replay):.4f}x")
        print(
            f"graph_with_{'exponential_race' if exponential_race else 'uniform'}_rng_us "
            f"mean={statistics.mean(stochastic_replay):.3f} "
            f"median={statistics.median(stochastic_replay):.3f}"
        )

    static_hidden = hidden.clone()
    static_frequencies = freq0.clone()
    static_allow_eos = allow_eos.clone()
    exact_graph = torch.npu.NPUGraph()
    exact_pool = torch.npu.graph_pool_handle()
    with torch.inference_mode(), torch.npu.graph(exact_graph, pool=exact_pool):
        graph_probabilities, graph_candidate_ids = filter_step(
            static_hidden,
            static_frequencies,
            weight,
            static_allow_eos,
            penalty,
        )
    second_graph = None
    second_hidden = None
    second_frequencies = None
    second_allow_eos = None
    second_probabilities = None
    second_candidate_ids = None
    if two_graphs:
        second_hidden = hidden.clone()
        second_frequencies = freq0.clone()
        second_allow_eos = torch.ones_like(allow_eos)
        second_graph = torch.npu.NPUGraph()
        with torch.inference_mode(), torch.npu.graph(second_graph, pool=exact_pool):
            second_probabilities, second_candidate_ids = filter_step(
                second_hidden,
                second_frequencies,
                weight,
                second_allow_eos,
                penalty,
            )
    print(
        "graph_output_formats "
        f"probabilities={torch_npu.get_npu_format(graph_probabilities)} "
        f"candidate_ids={torch_npu.get_npu_format(graph_candidate_ids)}"
    )
    if nd_boundary:
        consumer_probabilities = torch_npu.npu_format_cast(
            torch.empty_like(graph_probabilities),
            2,
        )
        consumer_candidate_ids = torch_npu.npu_format_cast(
            torch.empty_like(graph_candidate_ids),
            2,
        )
    else:
        consumer_probabilities = graph_probabilities
        consumer_candidate_ids = graph_candidate_ids
    print(
        "consumer_formats "
        f"probabilities={torch_npu.get_npu_format(consumer_probabilities)} "
        f"candidate_ids={torch_npu.get_npu_format(consumer_candidate_ids)}"
    )
    exact_generator = torch.Generator(device=device).manual_seed(42)
    graph_step_index = 0

    def exact_graph_step() -> None:
        nonlocal graph_step_index
        static_hidden.copy_(hidden)
        static_frequencies.copy_(freq0)
        static_allow_eos.copy_(allow_eos)
        if two_graphs and graph_step_index % 2:
            assert second_graph is not None
            assert second_hidden is not None
            assert second_frequencies is not None
            assert second_allow_eos is not None
            assert second_probabilities is not None
            assert second_candidate_ids is not None
            second_hidden.copy_(hidden)
            second_frequencies.copy_(freq0)
            second_allow_eos.copy_(allow_eos)
            second_graph.replay()
            active_probabilities = second_probabilities
            active_candidate_ids = second_candidate_ids
        else:
            exact_graph.replay()
            active_probabilities = graph_probabilities
            active_candidate_ids = graph_candidate_ids
        graph_step_index += 1
        if nd_boundary:
            consumer_probabilities.copy_(active_probabilities)
            consumer_candidate_ids.copy_(active_candidate_ids)
            active_probabilities = consumer_probabilities
            active_candidate_ids = consumer_candidate_ids
        sampled_position = torch.multinomial(
            active_probabilities,
            num_samples=1,
            generator=exact_generator,
        )
        sampled_exact = active_candidate_ids.gather(-1, sampled_position)
        freq1.copy_(advance_frequency(freq0, sampled_exact, expired, vocab_ids))

    exact_generator_eager = torch.Generator(device=device).manual_seed(42)

    def exact_eager_step() -> None:
        eager_probabilities, eager_candidate_ids = filter_step(
            hidden,
            freq0,
            weight,
            allow_eos,
            penalty,
        )
        sampled_position = torch.multinomial(
            eager_probabilities,
            num_samples=1,
            generator=exact_generator_eager,
        )
        sampled_exact = eager_candidate_ids.gather(-1, sampled_position)
        freq1.copy_(advance_frequency(freq0, sampled_exact, expired, vocab_ids))

    for _ in range(5):
        exact_graph_step()
        exact_eager_step()
    torch.npu.synchronize()
    exact_graph_times = event_us(exact_graph_step)
    exact_eager_times = event_us(exact_eager_step)
    print(
        "exact_eager_us "
        f"mean={statistics.mean(exact_eager_times):.3f} "
        f"median={statistics.median(exact_eager_times):.3f}"
    )
    print(
        "exact_graph_us "
        f"mean={statistics.mean(exact_graph_times):.3f} "
        f"median={statistics.median(exact_graph_times):.3f}"
    )
    print(
        "exact_speedup="
        f"{statistics.mean(exact_eager_times) / statistics.mean(exact_graph_times):.4f}x"
    )
    time.sleep(0.1)


if __name__ == "__main__":
    main()
