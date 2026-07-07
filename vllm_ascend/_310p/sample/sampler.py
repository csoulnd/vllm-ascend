#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
import vllm.envs as envs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.sampler import (
    DEFAULT_LOGPROBS_MODE,
    AscendSampler,
    AscendTopKTopPSampler,
)
from vllm_ascend.utils import global_stream, npu_stream_switch

_CPU_GENERATOR_CACHE_310P: dict[int, tuple[torch.Generator, int, int]] = {}


def _get_cpu_generator_310p(i: int, generator: torch.Generator) -> torch.Generator:
    seed = generator.initial_seed()
    cache_entry = _CPU_GENERATOR_CACHE_310P.get(i)
    if cache_entry is None or cache_entry[1] != id(generator) or cache_entry[2] != seed:
        cpu_generator = torch.Generator(device="cpu")
        # 310P CPU fallback must not read state from an NPU generator: that can
        # block in seeded sampling paths. The cached CPU generator advances
        # independently after being initialized from the same request seed.
        cpu_generator.manual_seed(seed)
        cache_entry = (cpu_generator, id(generator), seed)
        _CPU_GENERATOR_CACHE_310P[i] = cache_entry
    return cache_entry[0]


def _fill_cpu_exponential_310p(
    q_cpu: torch.Tensor,
    generators: dict[int, torch.Generator],
    has_draft_mask: torch.Tensor | None = None,
) -> None:
    """Fill a CPU tensor with exponential values for 310P stability."""
    if has_draft_mask is not None and has_draft_mask.device.type != "cpu":
        has_draft_mask = has_draft_mask.cpu()
    all_rows_seeded = len(generators) == q_cpu.shape[0] and set(generators) == set(
        range(q_cpu.shape[0])
    )
    if not all_rows_seeded:
        q_cpu.exponential_()
    if not generators:
        return
    for i, generator in generators.items():
        cpu_gen = _get_cpu_generator_310p(i, generator)
        if has_draft_mask is not None:
            if not bool(has_draft_mask[i]):
                continue
            q_cpu[i].exponential_(generator=cpu_gen)
        else:
            q_cpu[i].exponential_(generator=cpu_gen)


def fill_exponential_310p(
    q: torch.Tensor,
    generators: dict[int, torch.Generator],
    has_draft_mask: torch.Tensor | None = None,
) -> None:
    """Fill ``q`` with exponential values using CPU RNG for 310P stability."""
    with npu_stream_switch(global_stream()):
        q_cpu = q.cpu()
        _fill_cpu_exponential_310p(q_cpu, generators, has_draft_mask)
        q.copy_(q_cpu.to(q.device))
    torch.npu.current_stream().wait_stream(global_stream())


def _random_sample_310p(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """310P-specific random sampling with CPU exponential generation for q."""
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs).cpu()
        _fill_cpu_exponential_310p(q, generators)
        q = q.npu()
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


def generate_uniform_probs_310p(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Generate flattened uniform samples on CPU for 310P seeded rejection."""
    uniform_cpu = torch.empty(num_tokens, dtype=torch.float32, device="cpu", pin_memory=True)
    all_rows_seeded = len(generators) == len(num_draft_tokens) and set(generators) == set(
        range(len(num_draft_tokens))
    )
    if not all_rows_seeded:
        uniform_cpu.uniform_()

    start = 0
    for i, num_draft in enumerate(num_draft_tokens):
        end = start + num_draft
        if num_draft > 0 and i in generators:
            cpu_gen = _get_cpu_generator_310p(i, generators[i])
            uniform_cpu[start:end].uniform_(generator=cpu_gen)
        start = end

    return uniform_cpu.to(device, non_blocking=True)


class AscendTopKTopPSampler310(AscendTopKTopPSampler):
    def forward_native(self, logits, generators, k, p):
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)
        if get_ascend_config().enable_reduce_sample:
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = cand_logits.softmax(dim=-1, dtype=torch.float32)
            pos = _random_sample_310p(probs, generators)  # [B]

            next_token = cand_idx.gather(dim=1, index=pos.unsqueeze(1)).squeeze(1)  # [B]
            return next_token, logits_to_return
        else:
            logits = self.apply_top_k_top_p(logits, k, p)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return _random_sample_310p(probs, generators), logits_to_return


class AscendSampler310(AscendSampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler310(logprobs_mode=logprobs_mode)
