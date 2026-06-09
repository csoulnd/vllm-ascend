#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Thin re-exports for 310P unit tests. Runtime flat_ssm prefill lives in
# patch_gdn_attn (per-builder buffer pool) to keep graph-replay tensor refs stable.
#

from vllm_ascend.patch.worker.patch_gdn_attn import (
    _expand_spec_ssm_indices_cpu_for_query_len as expand_spec_ssm_indices_cpu_for_query_len,
    _flatten_spec_ssm_state_indices_cpu as flatten_spec_ssm_state_indices_cpu,
)

__all__ = [
    "expand_spec_ssm_indices_cpu_for_query_len",
    "flatten_spec_ssm_state_indices_cpu",
]
