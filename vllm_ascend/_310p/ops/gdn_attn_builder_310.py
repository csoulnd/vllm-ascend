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

import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, CommonAttentionMetadata

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)


class AscendGDNAttentionMetadataBuilder310(AscendGDNAttentionMetadataBuilder):
    use_full_cuda_graph: bool

    def _pad_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_batch_size: int,
    ) -> None:
        num_spec_decodes = attn_metadata.num_spec_decodes
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert spec_state_indices is not None
        assert spec_sequence_masks is not None
        assert spec_query_start_loc is not None
        assert num_accepted_tokens is not None

        self.spec_state_indices_tensor[:num_spec_decodes].copy_(
            spec_state_indices,
            non_blocking=True,
        )
        attn_metadata.spec_state_indices_tensor = self.spec_state_indices_tensor[:graph_batch_size]
        attn_metadata.spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

        self.spec_sequence_masks[:num_spec_decodes].copy_(
            spec_sequence_masks[:num_spec_decodes],
            non_blocking=True,
        )
        attn_metadata.spec_sequence_masks = self.spec_sequence_masks[:graph_batch_size]
        attn_metadata.spec_sequence_masks[num_spec_decodes:].fill_(False)

        assert attn_metadata.non_spec_token_indx is not None
        assert attn_metadata.spec_token_indx is not None
        non_spec_tokens = attn_metadata.non_spec_token_indx
        spec_tokens = attn_metadata.spec_token_indx
        self.non_spec_token_indx[: non_spec_tokens.size(0)].copy_(
            non_spec_tokens,
            non_blocking=True,
        )
        self.spec_token_indx[: spec_tokens.size(0)].copy_(
            spec_tokens,
            non_blocking=True,
        )
        attn_metadata.non_spec_token_indx = self.non_spec_token_indx[: non_spec_tokens.size(0)]
        attn_metadata.spec_token_indx = self.spec_token_indx[: spec_tokens.size(0)]

        self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
            spec_query_start_loc,
            non_blocking=True,
        )
        attn_metadata.spec_query_start_loc = self.spec_query_start_loc[: graph_batch_size + 1]
        query_padding = attn_metadata.spec_query_start_loc[num_spec_decodes + 1 :]
        if query_padding.numel() > 0:
            query_padding.copy_(
                spec_query_start_loc[-1].expand_as(query_padding),
                non_blocking=True,
            )

        self.num_accepted_tokens[:num_spec_decodes].copy_(
            num_accepted_tokens,
            non_blocking=True,
        )
        attn_metadata.num_accepted_tokens = self.num_accepted_tokens[:graph_batch_size]
        attn_metadata.num_accepted_tokens[num_spec_decodes:].fill_(0)

    def _pad_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_batch_size: int,
    ) -> None:
        num_decodes = attn_metadata.num_decodes
        state_indices = attn_metadata.non_spec_state_indices_tensor
        query_start_loc = attn_metadata.non_spec_query_start_loc
        assert state_indices is not None
        assert query_start_loc is not None

        self.non_spec_state_indices_tensor[:num_decodes].copy_(
            state_indices,
            non_blocking=True,
        )
        attn_metadata.non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[:graph_batch_size]
        attn_metadata.non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

        self.non_spec_query_start_loc[: num_decodes + 1].copy_(
            query_start_loc,
            non_blocking=True,
        )
        attn_metadata.non_spec_query_start_loc = self.non_spec_query_start_loc[: graph_batch_size + 1]
        query_padding = attn_metadata.non_spec_query_start_loc[num_decodes + 1 :]
        if query_padding.numel() > 0:
            query_padding.copy_(
                query_start_loc[-1].expand_as(query_padding),
                non_blocking=True,
            )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        use_full_graph = self.use_full_cuda_graph
        self.use_full_cuda_graph = False
        try:
            attn_metadata = super().build(
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                fast_build,
            )
        finally:
            self.use_full_cuda_graph = use_full_graph

        if not use_full_graph:
            return attn_metadata

        graph_batch_size = common_attn_metadata.num_reqs
        if (
            attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_spec_decodes <= self.decode_cudagraph_max_bs
            and attn_metadata.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self._pad_spec_decode_metadata(attn_metadata, graph_batch_size)
        elif (
            attn_metadata.num_prefills == 0
            and attn_metadata.num_spec_decodes == 0
            and attn_metadata.num_decodes <= self.decode_cudagraph_max_bs
        ):
            self._pad_decode_metadata(attn_metadata, graph_batch_size)
        return attn_metadata


class AscendGDNAttentionBackend310(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder310]:
        return AscendGDNAttentionMetadataBuilder310
