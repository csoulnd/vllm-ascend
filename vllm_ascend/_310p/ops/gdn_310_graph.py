#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""310P GDN ACL graph conv1d replay and device-side metadata helpers."""

import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.compilation.acl_graph import get_draft_graph_params, get_graph_params
from vllm_ascend.ops.gdn import (
    _check_and_get_host_args,
    _pad_conv1d_host_args_to_capture,
    get_causal_conv1d_update_host_args,
    get_spec_causal_conv1d_update_host_args,
)

_CONV1D_310_OP_BACKEND = "310"
_CONV1D_310_BUFFER_REPLAY = "buffer_replay"


def get_spec_causal_conv1d_device_args(
    attn_metadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fallback_meta = _check_and_get_host_args(attn_metadata, "spec_decode_fallback_meta", "spec_causal_conv1d")
    device_meta = fallback_meta.spec_causal_conv1d_device
    if device_meta is None:
        raise RuntimeError(
            "Expected attn_metadata.spec_decode_fallback_meta.spec_causal_conv1d_device for GDN spec conv1d path."
        )
    return (
        device_meta.query_start_loc,
        device_meta.cache_indices,
        device_meta.num_accepted_tokens,
        device_meta.query_start_loc_buffer,
        device_meta.cache_indices_buffer,
        device_meta.num_accepted_tokens_buffer,
    )


def get_non_spec_prefill_causal_conv1d_device_args(
    attn_metadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fallback_meta = _check_and_get_host_args(attn_metadata, "non_spec_prefill_fallback_meta", "causal_conv1d")
    device_meta = fallback_meta.causal_conv1d_device
    if device_meta is None or device_meta.has_initial_state is None:
        raise RuntimeError(
            "Expected attn_metadata.non_spec_prefill_fallback_meta.causal_conv1d_device "
            "for GDN non-spec prefill conv1d path."
        )
    return (
        device_meta.query_start_loc,
        device_meta.cache_indices,
        device_meta.has_initial_state,
        device_meta.query_start_loc_buffer,
        device_meta.cache_indices_buffer,
        device_meta.has_initial_state_buffer,
    )


def get_non_spec_decode_causal_conv1d_device_args(
    attn_metadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fallback_meta = _check_and_get_host_args(attn_metadata, "non_spec_decode_fallback_meta", "causal_conv1d")
    device_meta = fallback_meta.causal_conv1d_device
    if device_meta is None:
        raise RuntimeError(
            "Expected attn_metadata.non_spec_decode_fallback_meta.causal_conv1d_device "
            "for GDN non-spec decode conv1d path."
        )
    return (
        device_meta.query_start_loc,
        device_meta.cache_indices,
        device_meta.query_start_loc_buffer,
        device_meta.cache_indices_buffer,
    )


def copy_host_tuple_to_int64_buffer(
    buffer: torch.Tensor,
    host_tuple: tuple[int, ...],
) -> None:
    if not host_tuple:
        return
    num_elements = len(host_tuple)
    cpu_values = torch.tensor(host_tuple, dtype=torch.int64, device="cpu", pin_memory=buffer.is_pinned())
    buffer[:num_elements].copy_(cpu_values, non_blocking=True)


def update_conv1d_graph_params(
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    is_draft_model=False,
    draft_attn_metadatas=None,
):
    """Update device-side conv1d metadata buffers for 310P ACL graph replay."""
    graph_params = get_draft_graph_params() if is_draft_model else get_graph_params()

    if (
        graph_params is None
        or num_tokens not in graph_params.conv1d_params
        or len(graph_params.conv1d_params[num_tokens]) == 0
    ):
        return

    attn_metadata = forward_context.attn_metadata
    if is_draft_model and draft_attn_metadatas is not None:
        attn_metadata = draft_attn_metadatas

    with torch.npu.stream(update_stream):
        for param in graph_params.conv1d_params[num_tokens]:
            param_list = list(param)
            op_backend = param_list[14] if len(param_list) > 14 else "custom"
            replay_mode = param_list[15] if len(param_list) > 15 else "graph_task_update"
            if op_backend != _CONV1D_310_OP_BACKEND or replay_mode != _CONV1D_310_BUFFER_REPLAY:
                continue

            (
                _output,
                mixed_qkv,
                _conv_weights_T,
                _conv_state,
                _bias,
                _activation_num,
                _pad_slot_id,
                run_mode,
                branch,
                layer_prefix,
                qsl_dev,
                cidx_dev,
                nat_dev,
                q_per_seq,
            ) = param_list[:14]

            new_query_start_loc: tuple[int, ...] = ()
            new_cache_indices: tuple[int, ...] = ()
            new_num_accepted: tuple[int, ...] = ()

            if run_mode == 1 and attn_metadata is not None:
                meta = attn_metadata
                if isinstance(meta, dict):
                    meta = meta.get(layer_prefix, None)
                    assert isinstance(meta, GDNAttentionMetadata)

                if meta is None:
                    continue

                cap_x_dim0 = int(mixed_qkv.size(0))
                if branch == "spec" and meta.spec_sequence_masks is not None:
                    qsl_host, cidx_host, num_accepted_host = get_spec_causal_conv1d_update_host_args(meta)
                    new_query_start_loc, new_cache_indices, new_num_accepted = _pad_conv1d_host_args_to_capture(
                        qsl_host,
                        cidx_host,
                        num_accepted_host,
                        cap_x_dim0=cap_x_dim0,
                        q_per_seq=q_per_seq,
                        with_num_accepted=True,
                    )
                elif branch == "non_spec_decode":
                    non_sdq_host, non_sd_cidx_host = get_causal_conv1d_update_host_args(meta)
                    new_query_start_loc, new_cache_indices, _ = _pad_conv1d_host_args_to_capture(
                        non_sdq_host,
                        non_sd_cidx_host,
                        (),
                        cap_x_dim0=cap_x_dim0,
                        q_per_seq=q_per_seq,
                        with_num_accepted=False,
                    )
                    new_num_accepted = ()

            copy_host_tuple_to_int64_buffer(qsl_dev, new_query_start_loc)
            copy_host_tuple_to_int64_buffer(cidx_dev, new_cache_indices)
            if nat_dev is not None:
                copy_host_tuple_to_int64_buffer(nat_dev, new_num_accepted)
