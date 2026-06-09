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
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import get_draft_graph_params, get_graph_params
from vllm_ascend._310p.ops.gdn_310_graph import (
    get_non_spec_decode_causal_conv1d_device_args,
    get_non_spec_prefill_causal_conv1d_device_args,
    get_spec_causal_conv1d_device_args,
)
from vllm_ascend.utils import enable_sp, vllm_version_is, weak_ref_tensors

if vllm_version_is("0.20.2"):
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (  # type: ignore[import-not-found]
        GatedDeltaNetAttention,
    )
else:
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch
from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector

_CONV1D_310_OP_BACKEND = "310"
_CONV1D_310_BUFFER_REPLAY = "buffer_replay"


def _run_causal_conv1d_310(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    *,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    initial_state_mode: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    activation_num: int,
    run_mode: int,
    graph_params_key: int,
    branch: str,
    layer_prefix: str,
    qsl_buffer: torch.Tensor | None,
    cidx_buffer: torch.Tensor | None,
    nat_buffer: torch.Tensor | None,
    q_per_seq: int,
) -> torch.Tensor:
    if _EXTRA_CTX.capturing:
        assert qsl_buffer is not None and cidx_buffer is not None
        graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
        _register_310_conv1d_buffer_replay(
            graph_params,
            graph_params_key,
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            conv_state=conv_state,
            bias=bias,
            activation_num=activation_num,
            run_mode=run_mode,
            branch=branch,
            layer_prefix=layer_prefix,
            qsl_dev=qsl_buffer,
            cidx_dev=cidx_buffer,
            nat_dev=nat_buffer,
            q_per_seq=q_per_seq,
        )
    return torch.ops._C_ascend.npu_causal_conv1d_310(
        mixed_qkv,
        conv_weights,
        bias=bias,
        conv_states=conv_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=initial_state_mode,
        num_accepted_tokens=num_accepted_tokens,
        activation_mode=activation_num,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=run_mode,
    )


def _register_310_conv1d_buffer_replay(
    graph_params,
    num_actual_tokens: int,
    *,
    mixed_qkv,
    conv_weights,
    conv_state,
    bias,
    activation_num: int,
    run_mode: int,
    branch: str,
    layer_prefix: str,
    qsl_dev: torch.Tensor,
    cidx_dev: torch.Tensor,
    nat_dev: torch.Tensor | None,
    q_per_seq: int,
) -> None:
    graph_params.conv1d_params[num_actual_tokens].append(
        (
            None,
            weak_ref_tensors(mixed_qkv),
            weak_ref_tensors(conv_weights),
            weak_ref_tensors(conv_state),
            bias,
            activation_num,
            PAD_SLOT_ID,
            run_mode,
            branch,
            layer_prefix,
            weak_ref_tensors(qsl_dev),
            weak_ref_tensors(cidx_dev),
            weak_ref_tensors(nat_dev) if nat_dev is not None else None,
            q_per_seq,
            _CONV1D_310_OP_BACKEND,
            _CONV1D_310_BUFFER_REPLAY,
        )
    )
    graph_params.conv1d_handles[num_actual_tokens].append(None)
    graph_params.conv1d_events[num_actual_tokens].append(None)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x.to(torch.float32), p=2, dim=-1, eps=eps).to(x.dtype)


def _recurrent_token_count(q: torch.Tensor) -> int:
    if q.dim() == 4:
        return int(q.shape[1])
    if q.dim() == 3:
        return int(q.shape[0])
    raise ValueError(f"Unsupported recurrent q ndim={q.dim()}; expected 3D(TND) or 4D(BTND).")


def _ensure_recurrent_batch_dim(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if q.dim() == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    if g is not None and g.dim() == 2:
        g = g.unsqueeze(0)
    if beta.dim() == 2:
        beta = beta.unsqueeze(0)
    return q, k, v, g, beta


def _normalize_core_attn_out_b1td(
    core_attn_out: torch.Tensor,
    expected_tokens: int,
) -> torch.Tensor:
    if core_attn_out.dim() == 2:
        core_attn_out = core_attn_out.unsqueeze(0)
    elif core_attn_out.dim() >= 3 and core_attn_out.size(0) != 1:
        if core_attn_out.size(0) == expected_tokens:
            core_attn_out = core_attn_out.unsqueeze(0)
    if core_attn_out.size(0) != 1 or core_attn_out.size(1) != expected_tokens:
        raise RuntimeError(
            "GDN core attention output token count mismatch: "
            f"expected batch=1, tokens={expected_tokens}, got shape={tuple(core_attn_out.shape)}."
        )
    return core_attn_out.contiguous()


def _zero_rows_without_initial_state(
    initial_state: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    if has_initial_state.numel() != initial_state.shape[0]:
        raise ValueError(
            "has_initial_state size mismatch: "
            f"expected {initial_state.shape[0]}, got {has_initial_state.numel()}."
        )
    if has_initial_state.all():
        return initial_state
    mask = has_initial_state.to(initial_state.device).view(
        -1,
        *([1] * (initial_state.dim() - 1)),
    )
    return initial_state * mask.to(initial_state.dtype)


def _merge_spec_non_spec_core_attn_out(
    num_actual_tokens: int,
    spec_token_indx: torch.Tensor,
    core_attn_out_spec: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
    core_attn_out_non_spec: torch.Tensor,
) -> torch.Tensor:
    spec_out = _normalize_core_attn_out_b1td(
        core_attn_out_spec,
        int(spec_token_indx.numel()),
    )
    non_spec_out = _normalize_core_attn_out_b1td(
        core_attn_out_non_spec,
        int(non_spec_token_indx.numel()),
    )
    trailing_shape = spec_out.shape[2:]
    spec_flat = spec_out.reshape(spec_out.size(0), spec_out.size(1), -1)
    non_spec_flat = non_spec_out.reshape(non_spec_out.size(0), non_spec_out.size(1), -1)
    if spec_flat.shape[2] != non_spec_flat.shape[2]:
        raise RuntimeError(
            "GDN merge feature dim mismatch: "
            f"spec={tuple(spec_out.shape)}, non_spec={tuple(non_spec_out.shape)}."
        )

    spec_idx = spec_token_indx.to(dtype=torch.int64).contiguous()
    non_spec_idx = non_spec_token_indx.to(dtype=torch.int64).contiguous()

    # NPU index_copy_ can fail tiling on mixed spec/prefill batches; merge on CPU.
    merged_cpu = torch.zeros(
        1,
        num_actual_tokens,
        spec_flat.shape[2],
        dtype=spec_flat.dtype,
        device="cpu",
    )
    merged_cpu[0, spec_idx.cpu()] = spec_flat[0].detach().cpu()
    merged_cpu[0, non_spec_idx.cpu()] = non_spec_flat[0].detach().cpu()
    merged = merged_cpu.to(device=spec_out.device, non_blocking=False)
    return merged.reshape(1, num_actual_tokens, *trailing_shape)


def _flatten_state_indices(
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    if ssm_state_indices.ndim == 1:
        return ssm_state_indices[:total_tokens].to(torch.int32).contiguous()

    # masked_select on NPU triggers stream sync and breaks ACL graph capture.
    # Compact 2D indices on CPU, then copy back asynchronously.
    num_seqs = (cu_seqlens[1:] - cu_seqlens[:-1]).shape[0]
    ssm_cpu = ssm_state_indices[:num_seqs].cpu()
    seq_lens = cu_seqlens[1 : num_seqs + 1].cpu() - cu_seqlens[:num_seqs].cpu()
    q_per_seq = ssm_cpu.shape[1]
    positions = torch.arange(q_per_seq)
    valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    flat_cpu = ssm_cpu.masked_select(valid).to(torch.int32).contiguous()[:total_tokens]
    if not flat_cpu.is_pinned:
        flat_cpu = flat_cpu.pin_memory()
    flat_dev = torch.empty(
        flat_cpu.numel(),
        dtype=torch.int32,
        device=ssm_state_indices.device,
    )
    flat_dev.copy_(flat_cpu, non_blocking=True)
    return flat_dev.contiguous()


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
    flat_ssm_state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    q, k, v, g, beta = _ensure_recurrent_batch_dim(q, k, v, g, beta)
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    total_tokens = _recurrent_token_count(q)
    if flat_ssm_state_indices is not None:
        flat_state_indices = flat_ssm_state_indices[:total_tokens].to(torch.int32).contiguous()
    else:
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


def _310p_get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
    conv_state_dtype, _ = _original_get_state_dtype(self)
    return conv_state_dtype, torch.float16


_original_get_state_dtype = GatedDeltaNetAttention.get_state_dtype


class AscendGatedDeltaNetAttention310(GatedDeltaNetAttention):
    get_state_dtype = _310p_get_state_dtype

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
            qsl, cidx, nat, qsl_buf, cidx_buf, nat_buf = get_spec_causal_conv1d_device_args(attn_metadata)
            spec_q_per_seq = int(attn_metadata.spec_state_indices_tensor.size(-1))
            mixed_qkv_spec = _run_causal_conv1d_310(
                mixed_qkv_spec,
                conv_weights,
                self.conv1d.bias,
                conv_state,
                query_start_loc=qsl,
                cache_indices=cidx,
                initial_state_mode=None,
                num_accepted_tokens=nat,
                activation_num=activation_num,
                run_mode=1,
                graph_params_key=num_actual_tokens,
                branch="spec",
                layer_prefix=self.prefix,
                qsl_buffer=qsl_buf,
                cidx_buffer=cidx_buf,
                nat_buffer=nat_buf,
                q_per_seq=spec_q_per_seq,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                qsl, cidx, ism, qsl_buf, cidx_buf, ism_buf = get_non_spec_prefill_causal_conv1d_device_args(
                    attn_metadata
                )
                mixed_qkv_non_spec = _run_causal_conv1d_310(
                    mixed_qkv_non_spec,
                    conv_weights,
                    self.conv1d.bias,
                    conv_state,
                    query_start_loc=qsl,
                    cache_indices=cidx,
                    initial_state_mode=ism,
                    num_accepted_tokens=None,
                    activation_num=activation_num,
                    run_mode=0,
                    graph_params_key=num_actual_tokens,
                    branch="non_spec_prefill",
                    layer_prefix=self.prefix,
                    qsl_buffer=qsl_buf,
                    cidx_buffer=cidx_buf,
                    nat_buffer=ism_buf,
                    q_per_seq=1,
                )
        elif attn_metadata.num_decodes > 0:
            qsl, cidx, qsl_buf, cidx_buf = get_non_spec_decode_causal_conv1d_device_args(attn_metadata)
            mixed_qkv_non_spec = _run_causal_conv1d_310(
                mixed_qkv_non_spec,
                conv_weights,
                self.conv1d.bias,
                conv_state,
                query_start_loc=qsl,
                cache_indices=cidx,
                initial_state_mode=None,
                num_accepted_tokens=None,
                activation_num=activation_num,
                run_mode=1,
                graph_params_key=num_actual_tokens,
                branch="non_spec_decode",
                layer_prefix=self.prefix,
                qsl_buffer=qsl_buf,
                cidx_buffer=cidx_buf,
                nat_buffer=None,
                q_per_seq=1,
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
                spec_flat_ssm_state_indices = getattr(attn_metadata, "spec_flat_ssm_state_indices", None)
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
                    flat_ssm_state_indices=spec_flat_ssm_state_indices,
                )
            else:
                core_attn_out_spec = None

            # 2.2: Process the remaining part
            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                initial_state = _zero_rows_without_initial_state(initial_state, has_initial_state)
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
            merged_out = _merge_spec_non_spec_core_attn_out(
                num_actual_tokens,
                spec_token_indx,
                core_attn_out_spec,
                non_spec_token_indx,
                core_attn_out_non_spec,
            )
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
