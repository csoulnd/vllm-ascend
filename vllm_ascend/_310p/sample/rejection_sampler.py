#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from contextlib import contextmanager

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from vllm_ascend._310p.sample.sampler import (
    fill_exponential_310p,
    generate_uniform_probs_310p,
)
from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
)
from vllm_ascend.utils import global_stream, npu_stream_switch


@contextmanager
def _bind_310p_rejection_rng():
    """Route 310P rejection RNG (uniform / recovered-token q) through CPU."""
    original_sample_recovered = rejection_sampler_module.sample_recovered_tokens
    original_generate_uniform = rejection_sampler_module.generate_uniform_probs
    rejection_sampler_module.sample_recovered_tokens = _sample_recovered_tokens_310p
    rejection_sampler_module.generate_uniform_probs = _generate_uniform_probs_310p
    try:
        yield
    finally:
        rejection_sampler_module.sample_recovered_tokens = original_sample_recovered
        rejection_sampler_module.generate_uniform_probs = original_generate_uniform


def _generate_uniform_probs_310p(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """CPU uniform RNG, then blocking copy to NPU (same pattern as fill_exponential_310p)."""
    uniform_cpu = generate_uniform_probs_310p(
        num_tokens,
        num_draft_tokens,
        generators,
    )
    with npu_stream_switch(global_stream()):
        uniform_npu = torch.empty((num_tokens,), dtype=torch.float64, device=device)
        uniform_npu.copy_(uniform_cpu, non_blocking=False)
    torch.npu.current_stream().wait_stream(global_stream())
    return uniform_npu


def _sample_recovered_tokens_310p(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    use_block_verify: bool = False,
    target_indices: torch.Tensor | None = None,
    global_vocab_size: int | None = None,
    enable_reduce_sampling: bool = False,
) -> torch.Tensor:
    del global_vocab_size  # unused; kept for signature compatibility

    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]

    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
    has_draft_mask = num_draft_tensor > 0
    fill_exponential_310p(q, sampling_metadata.generators, has_draft_mask)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    if use_block_verify:
        sample_recovered_tokens_blockwise_pytorch(
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=draft_probs is None,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )
    else:
        sample_recovered_tokens_pytorch(
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=draft_probs is None,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )
    return recovered_token_ids


class AscendRejectionSampler310(AscendRejectionSampler):
    """310P rejection sampler: CPU RNG for uniform/recovered-token q; reject logic on NPU."""

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        with _bind_310p_rejection_rng():
            return super().forward(metadata, draft_probs, logits, sampling_metadata)
