#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


@dataclass
class _FlatSsmStateIndicesBufferSlot:
    flat_ssm_state_indices: torch.Tensor


_POOL: list[_FlatSsmStateIndicesBufferSlot] = []
_POOL_IDX = -1
_POOL_MAX_ELEMENTS = 0


def expand_spec_ssm_indices_cpu_for_query_len(
    ssm_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Expand rows when verify ``q_len`` exceeds ``spec_state_indices`` width.

    Upstream ``spec_state_indices_tensor`` is often ``[num_spec, num_spec]`` (draft
    slots) while MTP verify schedules ``1 + num_spec`` tokens per request. Without
    expansion, flatten yields too few indices and recurrent reads stale pool bytes
    for the remaining tokens (MTP=2: 2 indices per seq but 3 tokens).
    """
    if ssm_cpu.ndim == 1 or ssm_cpu.shape[1] == 0 or seq_lens.numel() == 0:
        return ssm_cpu

    q_per_seq = ssm_cpu.shape[1]
    max_seq_len = int(seq_lens.max().item())
    if max_seq_len <= q_per_seq:
        return ssm_cpu

    extra = max_seq_len - q_per_seq
    if extra == 1:
        # Leading target token reuses the first draft-slot index column.
        return torch.cat([ssm_cpu[:, :1], ssm_cpu], dim=1)[:, :max_seq_len]

    prefix = ssm_cpu[:, :1].expand(-1, extra)
    return torch.cat([prefix, ssm_cpu], dim=1)[:, :max_seq_len]


def flatten_spec_ssm_state_indices_cpu(
    spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_spec_decodes: int,
) -> torch.Tensor:
    ssm_cpu = spec_state_indices_tensor[:num_spec_decodes].cpu()
    if ssm_cpu.ndim == 1:
        return ssm_cpu.to(torch.int32).contiguous()

    cu_cpu = spec_query_start_loc[: num_spec_decodes + 1].cpu()
    seq_lens = cu_cpu[1:] - cu_cpu[:-1]
    ssm_cpu = expand_spec_ssm_indices_cpu_for_query_len(ssm_cpu, seq_lens)
    q_per_seq = ssm_cpu.shape[1]
    positions = torch.arange(q_per_seq)
    valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    return ssm_cpu.masked_select(valid).to(torch.int32).contiguous()


def max_flat_ssm_elements_from_builder(builder) -> int:
    """Match graph-capture upper bound: decode_cudagraph_max_bs * (1 + num_spec)."""
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    spec_cfg = builder.vllm_config.speculative_config
    num_speculative_tokens = spec_cfg.num_speculative_tokens if spec_cfg else 0
    decode_cudagraph_max_bs = getattr(builder, "decode_cudagraph_max_bs", max_num_seqs)
    return decode_cudagraph_max_bs * (num_speculative_tokens + 1)


def _ensure_flat_ssm_pool(device: torch.device, max_elements: int) -> None:
    global _POOL_MAX_ELEMENTS
    if device.type == "cpu":
        return
    if max_elements > _POOL_MAX_ELEMENTS:
        _POOL.clear()
        _POOL.extend(
            [
                _FlatSsmStateIndicesBufferSlot(
                    flat_ssm_state_indices=torch.empty(max_elements, dtype=torch.int32, device=device),
                ),
                _FlatSsmStateIndicesBufferSlot(
                    flat_ssm_state_indices=torch.empty(max_elements, dtype=torch.int32, device=device),
                ),
            ]
        )
        _POOL_MAX_ELEMENTS = max_elements


def _acquire_flat_ssm_slot() -> _FlatSsmStateIndicesBufferSlot:
    global _POOL_IDX
    _POOL_IDX = (_POOL_IDX + 1) % len(_POOL)
    return _POOL[_POOL_IDX]


def fill_spec_flat_ssm_state_indices(
    attn_metadata: GDNAttentionMetadata,
    *,
    max_elements: int,
) -> None:
    """Pre-fill flattened SSM indices on CPU/NPU before graph replay (no D2H in forward)."""
    if attn_metadata.spec_state_indices_tensor is None or attn_metadata.spec_query_start_loc is None:
        attn_metadata.spec_flat_ssm_state_indices = None
        return

    flat_cpu = flatten_spec_ssm_state_indices_cpu(
        attn_metadata.spec_state_indices_tensor,
        attn_metadata.spec_query_start_loc,
        attn_metadata.num_spec_decodes,
    )
    if attn_metadata.spec_state_indices_tensor.device.type == "cpu":
        attn_metadata.spec_flat_ssm_state_indices = flat_cpu
        return

    device = attn_metadata.spec_state_indices_tensor.device
    _ensure_flat_ssm_pool(device, max_elements)

    slot = _acquire_flat_ssm_slot()
    num_elements = flat_cpu.numel()
    if num_elements > max_elements:
        raise RuntimeError(
            f"Flattened SSM indices ({num_elements}) exceed graph buffer "
            f"({max_elements}); check MTP spec_state_indices layout."
        )
    if not flat_cpu.is_pinned:
        flat_cpu = flat_cpu.pin_memory()
    slot.flat_ssm_state_indices[:num_elements].copy_(flat_cpu, non_blocking=True)
    attn_metadata.spec_flat_ssm_state_indices = slot.flat_ssm_state_indices


def fill_spec_flat_ssm_state_indices_for_builder(builder, attn_metadata: GDNAttentionMetadata) -> None:
    """Called from GDN metadata builder on 310P during spec-decode metadata build."""
    fill_spec_flat_ssm_state_indices(
        attn_metadata,
        max_elements=max_flat_ssm_elements_from_builder(builder),
    )
