# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend._310p.ops.fla import gdn_310


def test_recurrent_token_count_uses_token_dim_for_tnd():
    q_tnd = torch.zeros(2, 4, 8)
    assert gdn_310._recurrent_token_count(q_tnd) == 2

    q_btnd = torch.zeros(1, 2, 4, 8)
    assert gdn_310._recurrent_token_count(q_btnd) == 2


def test_normalize_core_attn_out_b1td_promotes_tnd():
    out = torch.zeros(73, 4, 8)
    normalized = gdn_310._normalize_core_attn_out_b1td(out, 73)
    assert normalized.shape == (1, 73, 4, 8)


def test_merge_spec_non_spec_core_attn_out():
    spec_idx = torch.tensor([73, 74], dtype=torch.int64)
    non_spec_idx = torch.arange(73, dtype=torch.int64)
    spec_out = torch.full((1, 2, 4, 8), 1.0)
    non_spec_out = torch.full((73, 4, 8), 2.0)

    merged = gdn_310._merge_spec_non_spec_core_attn_out(
        75,
        spec_idx,
        spec_out,
        non_spec_idx,
        non_spec_out,
    )
    assert merged.shape == (1, 75, 4, 8)
    assert torch.all(merged[0, :73] == 2.0)
    assert torch.all(merged[0, 73:] == 1.0)


def test_zero_rows_without_initial_state_avoids_boolean_index_put():
    initial_state = torch.ones(2, 3, 4)
    has_initial_state = torch.tensor([True, False])
    cleared = gdn_310._zero_rows_without_initial_state(initial_state, has_initial_state)
    assert torch.all(cleared[0] == 1.0)
    assert torch.all(cleared[1] == 0.0)


def test_zero_rows_without_initial_state_rejects_size_mismatch():
    initial_state = torch.ones(2, 3, 4)
    has_initial_state = torch.tensor([True])
    with pytest.raises(ValueError, match="size mismatch"):
        gdn_310._zero_rows_without_initial_state(initial_state, has_initial_state)
