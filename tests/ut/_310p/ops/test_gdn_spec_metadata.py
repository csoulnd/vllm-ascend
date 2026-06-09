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

from vllm_ascend._310p.ops.fla.gdn_spec_metadata import flatten_spec_ssm_state_indices_cpu


def test_flatten_spec_ssm_state_indices_expands_mtp_verify_query_len():
    """MTP verify q_len=1+num_spec must flatten to one index per token."""
    ssm = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 6], dtype=torch.int32)
    flat = flatten_spec_ssm_state_indices_cpu(ssm, qsl, num_spec_decodes=2)
    assert flat.tolist() == [10, 10, 11, 20, 20, 21]


def test_flatten_spec_ssm_state_indices_unchanged_when_width_matches():
    ssm = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 6], dtype=torch.int32)
    flat = flatten_spec_ssm_state_indices_cpu(ssm, qsl, num_spec_decodes=2)
    assert flat.tolist() == [10, 11, 12, 20, 21, 22]
