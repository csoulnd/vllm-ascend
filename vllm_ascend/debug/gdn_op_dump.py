# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Dump GDN spec-path operator I/O for MTP first-verify debugging."""

from __future__ import annotations

import os
import re
from typing import Any

import torch

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import is_310p

_DUMP_DONE = False


def _to_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu()


def _target_step() -> int:
    raw = envs.VLLM_ASCEND_GDN_DUMP_STEP
    return int(raw.strip())


def _target_layer() -> int:
    return int(envs.VLLM_ASCEND_GDN_DUMP_LAYER)


def parse_layer_index(prefix: str) -> int | None:
    match = re.search(r"layers\.(\d+)", prefix)
    if match is None:
        return None
    return int(match.group(1))


def current_forward_step() -> int:
    try:
        return int(_EXTRA_CTX.mtp_forward_step)
    except AttributeError:
        return 0


def should_dump_gdn_spec_ops(prefix: str, attn_metadata: Any) -> bool:
    """True only for first-verify spec-only forward on the configured layer."""
    global _DUMP_DONE
    if not envs.VLLM_ASCEND_GDN_DUMP or _DUMP_DONE:
        return False
    if getattr(attn_metadata, "spec_sequence_masks", None) is None:
        return False
    # MTP first verify: spec tokens only (no interleaved prefill/decode in this pass).
    if attn_metadata.num_prefills != 0 or attn_metadata.num_decodes != 0:
        return False
    if current_forward_step() != _target_step():
        return False
    layer_idx = parse_layer_index(prefix)
    if layer_idx is None or layer_idx != _target_layer():
        return False
    return True


def mark_gdn_dump_done() -> None:
    global _DUMP_DONE
    _DUMP_DONE = True


def save_gdn_op_dump(
    *,
    layer_prefix: str,
    op_name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> str:
    dump_dir = envs.VLLM_ASCEND_GDN_DUMP_DIR
    os.makedirs(dump_dir, exist_ok=True)
    path_tag = "310p" if is_310p() else "910"
    layer_idx = parse_layer_index(layer_prefix)
    step = current_forward_step()
    record = {
        "step": step,
        "path": path_tag,
        "layer_prefix": layer_prefix,
        "layer_index": layer_idx,
        "op": op_name,
        "inputs": {k: _to_cpu(v) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
        "outputs": {k: _to_cpu(v) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()},
    }
    dump_path = os.path.join(
        dump_dir,
        f"gdn_step{step:04d}_L{layer_idx}_{op_name}_{path_tag}.pt",
    )
    torch.save(record, dump_path)
    return dump_path
