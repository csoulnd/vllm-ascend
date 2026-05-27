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
# from collections.abc import Iterable
# mypy: ignore-errors


import torch
import torch.nn.functional as F
import torch_npu
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch
from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.compilation.acl_graph import get_draft_graph_params, get_graph_params
from vllm_ascend.ops.gdn import (
    _pad_conv1d_host_args_to_capture,
    get_causal_conv1d_update_host_args,
    get_non_spec_causal_conv1d_host_args,
    get_spec_causal_conv1d_update_host_args,
    to_int64_tuple,
)
from vllm_ascend.utils import enable_sp, weak_ref_tensors


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x.to(torch.float32), p=2, dim=-1, eps=eps).to(x.dtype)


def _flatten_state_indices(
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    if ssm_state_indices.ndim == 1:
        return ssm_state_indices[:total_tokens].to(torch.int32).contiguous()

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    ssm_state_indices = ssm_state_indices[: seq_lens.shape[0]]
    positions = torch.arange(
        ssm_state_indices.shape[1],
        device=ssm_state_indices.device,
        dtype=seq_lens.dtype,
    )
    valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    return ssm_state_indices.masked_select(valid)[:total_tokens].to(torch.int32).contiguous()


def npu_recurrent_gated_delta_rule_310(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    total_tokens = v.shape[1]
    flat_state_indices = _flatten_state_indices(ssm_state_indices, cu_seqlens, total_tokens)
    actual_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32).contiguous()
    accepted_tokens = None
    if num_accepted_tokens is not None:
        accepted_tokens = num_accepted_tokens[: actual_seq_lengths.shape[0]].to(torch.int32).contiguous()

    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=q.squeeze(0).to(torch.float16).contiguous(),
        key=k.squeeze(0).to(torch.float16).contiguous(),
        value=v.squeeze(0).to(torch.float16).contiguous(),
        g=None if g is None else g.squeeze(0).to(torch.float32).contiguous(),
        gk=None,
        beta=beta.squeeze(0).to(torch.float16).contiguous(),
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=flat_state_indices,
        num_accepted_tokens=accepted_tokens,
        scale_value=k.shape[-1] ** -0.5,
    ).unsqueeze(0)
    return out


def _run_causal_conv1d_310_custom(
    output: torch.Tensor,
    x: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_state: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc_opt: tuple[int, ...],
    cache_indices_opt: tuple[int, ...],
    initial_state_mode_opt: tuple[int, ...],
    num_accepted_tokens_opt: tuple[int, ...],
    activation_mode: int,
    run_mode: int,
) -> torch.Tensor:
    torch.ops._C_ascend.npu_causal_conv1d_310_custom(
        output,
        x,
        conv_weights,
        conv_states=conv_state,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc_opt,
        cache_indices_opt=cache_indices_opt,
        initial_state_mode_opt=initial_state_mode_opt,
        num_accepted_tokens_opt=num_accepted_tokens_opt,
        activation_mode=activation_mode,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=run_mode,
    )
    return output


def _capture_causal_conv1d_310(
    *,
    output: torch.Tensor,
    x: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_state: torch.Tensor,
    bias: torch.Tensor | None,
    activation_mode: int,
    run_mode: int,
    branch: str,
    layer_prefix: str,
    num_actual_tokens: int,
    query_start_loc_opt: tuple[int, ...],
    cache_indices_opt: tuple[int, ...],
    initial_state_mode_opt: tuple[int, ...],
    num_accepted_tokens_opt: tuple[int, ...],
    q_per_seq: int,
) -> torch.Tensor:
    stream = torch_npu.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
    graph_params.conv1d_events[num_actual_tokens].append(event)
    graph_params.conv1d_params[num_actual_tokens].append(
        (
            weak_ref_tensors(output),
            weak_ref_tensors(x),
            weak_ref_tensors(conv_weights),
            weak_ref_tensors(conv_state),
            bias,
            activation_mode,
            PAD_SLOT_ID,
            run_mode,
            branch,
            layer_prefix,
            query_start_loc_opt,
            cache_indices_opt,
            initial_state_mode_opt,
            num_accepted_tokens_opt,
            q_per_seq,
        )
    )

    torch.npu.graph_task_group_begin(stream)
    _run_causal_conv1d_310_custom(
        output,
        x,
        conv_weights,
        conv_state,
        bias,
        query_start_loc_opt,
        cache_indices_opt,
        initial_state_mode_opt,
        num_accepted_tokens_opt,
        activation_mode,
        run_mode,
    )
    handle = torch.npu.graph_task_group_end(stream)
    graph_params.conv1d_handles[num_actual_tokens].append(handle)
    return output


def update_conv1d_310_graph_params(
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    is_draft_model=False,
    draft_attn_metadatas=None,
):
    """Update host-side parameters for causal_conv1d_310 graph replay."""
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
        for param, handle, event in zip(
            graph_params.conv1d_params[num_tokens],
            graph_params.conv1d_handles[num_tokens],
            graph_params.conv1d_events[num_tokens],
        ):
            (
                output,
                mixed_qkv,
                conv_weights,
                conv_state,
                bias,
                activation_mode,
                pad_slot_id,
                run_mode,
                branch,
                layer_prefix,
                _,
                _,
                _,
                _,
                q_per_seq,
            ) = param

            new_query_start_loc: tuple[int, ...] = ()
            new_cache_indices: tuple[int, ...] = ()
            new_initial_state_mode: tuple[int, ...] = ()
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

            torch.npu.graph_task_update_begin(update_stream, handle)
            _run_causal_conv1d_310_custom(
                output,
                mixed_qkv,
                conv_weights,
                conv_state,
                bias,
                new_query_start_loc,
                new_cache_indices,
                new_initial_state_mode,
                new_num_accepted,
                activation_mode,
                run_mode,
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)


class AscendGatedDeltaNetAttention310(GatedDeltaNetAttention):
    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        conv_state_dtype, _ = super().get_state_dtype()
        return conv_state_dtype, torch.float16

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        # Core attention computation (called by custom op).

        # NOTE: The processing logic of Qwen3_5GatedDeltaNet is the same as Qwen3NextGatedDeltaNet.
        # However, because the ops `torch_npu.npu_recurrent_gated_delta_rule`
        # currently does not support `ssm_state` inputs in float32 format,
        # we temporarily retain the current _forward_core implementation.
        # Once the ops supports float32 `ssm_state`, this patch should be removed.

        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0]
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2)).transpose(0, 1)
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv
        activation_num = 1 if self.activation else 0

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            spec_qsl_host, spec_ci_host, spec_nat_host = get_spec_causal_conv1d_update_host_args(attn_metadata)
            if _EXTRA_CTX.capturing:
                output_spec = torch.empty_like(mixed_qkv_spec)
                spec_q_per_seq = int(attn_metadata.spec_state_indices_tensor.size(-1))
                mixed_qkv_spec = _capture_causal_conv1d_310(
                    output=output_spec,
                    x=mixed_qkv_spec,
                    conv_weights=conv_weights,
                    conv_state=conv_state,
                    bias=self.conv1d.bias,
                    activation_mode=activation_num,
                    run_mode=1,
                    branch="spec",
                    layer_prefix=self.prefix,
                    num_actual_tokens=num_actual_tokens,
                    query_start_loc_opt=spec_qsl_host,
                    cache_indices_opt=spec_ci_host,
                    initial_state_mode_opt=(),
                    num_accepted_tokens_opt=spec_nat_host,
                    q_per_seq=spec_q_per_seq,
                )
            else:
                output_spec = torch.empty_like(mixed_qkv_spec)
                mixed_qkv_spec = _run_causal_conv1d_310_custom(
                    output_spec,
                    mixed_qkv_spec,
                    conv_weights,
                    conv_state,
                    self.conv1d.bias,
                    spec_qsl_host,
                    spec_ci_host,
                    (),
                    spec_nat_host,
                    activation_num,
                    1,
                )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                activation_num = 1 if self.activation else 0
                query_start_loc_opt, cache_indices_opt, initial_state_mode_opt = get_non_spec_causal_conv1d_host_args(
                    attn_metadata
                )
                mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                mixed_qkv_non_spec = _run_causal_conv1d_310_custom(
                    mixed_qkv_non_spec_output,
                    mixed_qkv_non_spec,
                    conv_weights,
                    conv_state,
                    self.conv1d.bias,
                    query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    (),
                    activation_num,
                    0,
                )
        elif attn_metadata.num_decodes > 0:
            activation_num = 1 if self.activation else 0
            num_decodes = attn_metadata.num_decodes
            try:
                non_spec_qsl_host, non_spec_ci_host = get_causal_conv1d_update_host_args(attn_metadata)
            except RuntimeError:
                non_spec_qsl_host = to_int64_tuple(non_spec_query_start_loc[: num_decodes + 1])
                non_spec_ci_host = to_int64_tuple(non_spec_state_indices_tensor[:num_decodes])
            if _EXTRA_CTX.capturing:
                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
                mixed_qkv_non_spec = _capture_causal_conv1d_310(
                    output=output_non_spec,
                    x=mixed_qkv_non_spec,
                    conv_weights=conv_weights,
                    conv_state=conv_state,
                    bias=self.conv1d.bias,
                    activation_mode=activation_num,
                    run_mode=1,
                    branch="non_spec_decode",
                    layer_prefix=self.prefix,
                    num_actual_tokens=num_actual_tokens,
                    query_start_loc_opt=non_spec_qsl_host,
                    cache_indices_opt=non_spec_ci_host,
                    initial_state_mode_opt=(),
                    num_accepted_tokens_opt=(),
                    q_per_seq=1,
                )
            else:
                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
                mixed_qkv_non_spec = _run_causal_conv1d_310_custom(
                    output_non_spec,
                    mixed_qkv_non_spec,
                    conv_weights,
                    conv_state,
                    self.conv1d.bias,
                    non_spec_qsl_host,
                    non_spec_ci_host,
                    (),
                    (),
                    activation_num,
                    1,
                )
        else:
            mixed_qkv_non_spec = None
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        g, beta = fused_gdn_gating_pytorch(self.A_log, a, b, self.dt_bias)
        if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            # 2. Recurrent attention

            # 2.1: Process the multi-query part
            if spec_sequence_masks is not None:
                core_attn_out_spec = npu_recurrent_gated_delta_rule_310(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    state=ssm_state,
                    cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_spec = None

            # 2.2: Process the remaining part
            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                initial_state[~has_initial_state, ...] = 0
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_gated_delta_rule_pytorch(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )

                # Init cache
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
            elif attn_metadata.num_decodes > 0:
                core_attn_out_non_spec = npu_recurrent_gated_delta_rule_310(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    state=ssm_state,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_non_spec = None

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec = npu_recurrent_gated_delta_rule_310(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g,
                beta=beta,
                state=ssm_state,
                cu_seqlens=non_spec_query_start_loc,
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]
        maybe_save_kv_layer_to_connector("", [])
