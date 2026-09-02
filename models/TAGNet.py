import torch
import torch.nn as nn

from layers.Embed import ChannelBranchEmbedding, Frequency_Embedding, Multi_Resolution_Data
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import (
    CentralizedMixerLayer,
    FormerLayer,
    ResolutionRouter,
)
from layers.Multi_Resolution_GNN import MRGNN
from layers.Difference_Pre import (
    DataRestoration,
    DifferenceDataEmb,
    DifferenceTimeExpert,
)

class ChannelAwareResolutionEncoder(nn.Module):
    def __init__(self, configs, res_num, seq_len, enc_in, d_model, d_ff, dropout):
        super(ChannelAwareResolutionEncoder, self).__init__()
        self.res_num = res_num
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        disable_post_router_and_channel_injection = getattr(
            configs, "disable_post_router_and_channel_injection", False
        )
        self.use_channel_branch = getattr(configs, "use_channel_branch", True)
        self.use_channel_branch = self.use_channel_branch and not getattr(
            configs, "disable_channel_branch", False
        )
        self.use_resolution_router = getattr(configs, "use_resolution_router", True)
        self.use_resolution_router = self.use_resolution_router and not getattr(
            configs, "disable_resolution_router", False
        )
        self.use_post_resolution_router = not getattr(
            configs, "disable_post_resolution_router", False
        )
        self.use_post_resolution_router = (
            self.use_post_resolution_router
            and not disable_post_router_and_channel_injection
        )
        disable_post_channel_injection = getattr(
            configs, "disable_post_channel_injection", False
        ) or getattr(configs, "disable_channel_context_injection", False)
        self.use_channel_context_injection = (
            not disable_post_channel_injection
            and not disable_post_router_and_channel_injection
        )
        self.enable_joint_module = not getattr(
            configs, "disable_channel_resolution_module", False
        )

        self.channel_layers = getattr(configs, "channel_layers", 2)
        self.augmentations = configs.augmentations.split(",")
        self.channel_branch_embed = ChannelBranchEmbedding(
            seq_len,
            enc_in,
            d_model,
            dropout=dropout,
            augmentation=self.augmentations,
        )
        self.channel_branch_layers = nn.ModuleList(
            [
                CentralizedMixerLayer(
                    d_model,
                    d_ff=max(d_model * 2, d_ff // 2),
                    dropout=dropout,
                )
                for _ in range(self.channel_layers)
            ]
        )
        self.channel_fusion = nn.ModuleList(
            [nn.Linear(d_model * 2, d_model) for _ in range(res_num)]
        )
        self.pre_resolution_router = ResolutionRouter(d_model, dropout=dropout)
        self.post_resolution_router = ResolutionRouter(d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FormerLayer(
                        res_num,
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.output_gate = nn.Linear(d_model * 2, d_model)

        for fusion in self.channel_fusion:
            nn.init.constant_(fusion.bias, -1.0)
        nn.init.constant_(self.output_gate.bias, -1.0)

    def _encode_channel_branch(self, x_enc):
        channel_tokens = self.channel_branch_embed(x_enc)
        for layer in self.channel_branch_layers:
            channel_tokens = layer(channel_tokens)
        return channel_tokens

    def _fuse_channel_branch(self, base_enc, channel_tokens):
        fused_list = []
        for idx, res_token in enumerate(base_enc):
            gate = torch.sigmoid(
                self.channel_fusion[idx](torch.cat([res_token, channel_tokens], dim=-1))
            )
            fused_list.append(res_token + gate * (channel_tokens - res_token))
        return fused_list

    def _inject_channel_context(self, branch_tokens, channel_tokens):
        if channel_tokens is None:
            return branch_tokens

        fused_out = []
        for res_token in branch_tokens:
            gate = torch.sigmoid(
                self.output_gate(torch.cat([res_token, channel_tokens], dim=-1))
            )
            fused_out.append(res_token + gate * channel_tokens)
        return fused_out

    def _plain_encode(self, branch_tokens):
        encoded_tokens, _ = self.encoder(branch_tokens, attn_mask=None)
        return encoded_tokens

    def forward(self, branch_tokens, x_enc):
        if not self.enable_joint_module:
            return self._plain_encode(branch_tokens)

        channel_tokens = (
            self._encode_channel_branch(x_enc) if self.use_channel_branch else None
        )

        if channel_tokens is not None:
            branch_tokens = self._fuse_channel_branch(branch_tokens, channel_tokens)

        if self.use_resolution_router:
            branch_tokens, _ = self.pre_resolution_router(branch_tokens)

        branch_tokens, _ = self.encoder(branch_tokens, attn_mask=None)

        if self.use_resolution_router and self.use_post_resolution_router:
            branch_tokens, _ = self.post_resolution_router(branch_tokens)

        if self.use_channel_context_injection:
            branch_tokens = self._inject_channel_context(branch_tokens, channel_tokens)

        return branch_tokens


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.output_attention = configs.output_attention
        self.activation = configs.activation
        self.resolution_list = list(map(int, configs.resolution_list.split(",")))

        self.res_num = len(self.resolution_list)
        self.stride_list = self.resolution_list
        self.res_len = [int(self.seq_len // res) + 1 for res in self.resolution_list]
        self.augmentations = configs.augmentations.split(",")
        self.use_cross_graph_interaction = not getattr(
            configs, "disable_cross_graph_interaction", False
        )
        self.use_frequency_branch = not getattr(
            configs, "disable_frequency_branch", False
        )
        self.use_temporal_branch = not getattr(
            configs, "disable_temporal_branch", False
        )
        self.use_structure_alignment = not getattr(
            configs, "disable_structure_alignment", False
        )
        self.single_gnn = getattr(configs, "single_gnn", False)
        self.single_gnn_fusion = getattr(configs, "single_gnn_fusion", "gated")
        self.structure_delta_scale = getattr(configs, "structure_delta_scale", 0.1)
        self.low_freq_ratio = getattr(configs, "low_freq_ratio", 0.5)
        self.time_small_kernel = getattr(configs, "time_small_kernel", 3)
        self.time_large_kernel = getattr(configs, "time_large_kernel", 9)
        self.time_mamba_state = getattr(configs, "time_mamba_state", 16)
        self.time_mamba_conv = getattr(configs, "time_mamba_conv", 4)
        self.time_mamba_expand = getattr(configs, "time_mamba_expand", 2)
        self.time_ffn_expansion = getattr(configs, "time_ffn_expansion", 2.0)
        self.multi_res_data = Multi_Resolution_Data(
            self.enc_in, self.resolution_list, self.stride_list
        )
        self.freq_embedding = Frequency_Embedding(
            self.d_model,
            self.res_len,
            self.augmentations,
            dropout=self.dropout,
            low_freq_ratio=self.low_freq_ratio,
        )

        self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)
        self.difference_attention = DifferenceTimeExpert(
            self.res_num,
            self.d_model,
            small_kernel=self.time_small_kernel,
            large_kernel=self.time_large_kernel,
            d_state=self.time_mamba_state,
            d_conv=self.time_mamba_conv,
            expand=self.time_mamba_expand,
            dropout=self.dropout,
            ffn_expansion=self.time_ffn_expansion,
        )
        self.data_restoration = DataRestoration(
            self.res_num, self.enc_in, self.d_model
        )
        self.embeddings = nn.ModuleList(
            [nn.Linear(res_len, self.d_model) for res_len in self.res_len]
        )
        self.channel_resolution_encoder = ChannelAwareResolutionEncoder(
            configs,
            self.res_num,
            self.seq_len,
            self.enc_in,
            self.d_model,
            self.d_ff,
            self.dropout,
        )
        if self.single_gnn:
            self.shared_mrgnn = MRGNN(configs, self.res_len)
        else:
            self.freq_mrgnn = MRGNN(configs, self.res_len)
            self.diff_mrgnn = MRGNN(configs, self.res_len)
        self.cross_graph_freq = nn.ModuleList(
            [nn.Linear(self.d_model * 2, self.d_model) for _ in range(self.res_num)]
        )
        self.cross_graph_diff = nn.ModuleList(
            [nn.Linear(self.d_model * 2, self.d_model) for _ in range(self.res_num)]
        )
        self.structure_align_norm = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.res_num)]
        )
        self.graph_fusion_gate = nn.Linear(self.d_model * 2, self.d_model)
        self.single_concat_fusion = nn.Linear(self.d_model * 2, self.d_model)
        self.single_hint_freq = nn.Linear(self.d_model, self.d_model)
        self.single_hint_temporal = nn.Linear(self.d_model, self.d_model)
        self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)

        for layer in self.cross_graph_freq:
            nn.init.constant_(layer.bias, -1.0)
        for layer in self.cross_graph_diff:
            nn.init.constant_(layer.bias, -1.0)
        nn.init.constant_(self.graph_fusion_gate.bias, -1.0)

    def _build_temporal_tokens(self, multi_res_data):
        temporal_tokens, x_padding = self.diff_data_emb(multi_res_data)
        temporal_tokens, _ = self.difference_attention(temporal_tokens, attn_mask=None)
        temporal_tokens = self.data_restoration(temporal_tokens, x_padding)
        temporal_tokens = [
            self.embeddings[idx](temporal_tokens[idx]) for idx in range(self.res_num)
        ]
        return temporal_tokens

    def _fuse_branch_tokens(self, freq_tokens, temporal_tokens):
        fused_tokens = []
        for freq_token, temporal_token in zip(freq_tokens, temporal_tokens):
            if self.single_gnn_fusion == "add":
                fused_tokens.append(0.5 * (freq_token + temporal_token))
            elif self.single_gnn_fusion == "concat":
                fused_tokens.append(
                    self.single_concat_fusion(
                        torch.cat([freq_token, temporal_token], dim=-1)
                    )
                )
            elif self.single_gnn_fusion == "hint":
                freq_hint = freq_token + self.single_hint_freq(temporal_token)
                temporal_hint = temporal_token + self.single_hint_temporal(freq_token)
                fused_tokens.append(0.5 * (freq_hint + temporal_hint))
            else:
                gate = torch.sigmoid(
                    self.graph_fusion_gate(
                        torch.cat([freq_token, temporal_token], dim=-1)
                    )
                )
                fused_tokens.append(gate * freq_token + (1.0 - gate) * temporal_token)
        return fused_tokens

    def _cross_graph_interaction(self, freq_graph_list, diff_graph_list):
        interacted_freq = []
        interacted_diff = []
        for idx, (freq_feat, diff_feat) in enumerate(zip(freq_graph_list, diff_graph_list)):
            freq_context = diff_feat.permute(0, 2, 1)
            diff_context = freq_feat.permute(0, 2, 1)

            freq_gate = torch.sigmoid(
                self.cross_graph_freq[idx](torch.cat([freq_context, diff_context], dim=-1))
            ).permute(0, 2, 1)
            diff_gate = torch.sigmoid(
                self.cross_graph_diff[idx](torch.cat([diff_context, freq_context], dim=-1))
            ).permute(0, 2, 1)

            interacted_freq.append(freq_feat + freq_gate * diff_feat)
            interacted_diff.append(diff_feat + diff_gate * freq_feat)

        return interacted_freq, interacted_diff

    def _normalize_adjacency(self, adjacency):
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _apply_structure_alignment(
        self, freq_graph_list, diff_graph_list, freq_adj_list, diff_adj_list
    ):
        aligned_freq = []
        aligned_diff = []

        for idx, (freq_feat, diff_feat, freq_adj, diff_adj) in enumerate(
            zip(freq_graph_list, diff_graph_list, freq_adj_list, diff_adj_list)
        ):
            a_common = self._normalize_adjacency(0.5 * (freq_adj + diff_adj))
            a_delta = freq_adj - diff_adj

            freq_struct = self._normalize_adjacency(
                a_common + self.structure_delta_scale * a_delta
            )
            diff_struct = self._normalize_adjacency(
                a_common - self.structure_delta_scale * a_delta
            )

            freq_nodes = freq_feat.permute(0, 2, 1)
            diff_nodes = diff_feat.permute(0, 2, 1)

            freq_aligned = torch.einsum("bij,bjd->bid", freq_struct, freq_nodes)
            diff_aligned = torch.einsum("bij,bjd->bid", diff_struct, diff_nodes)

            freq_aligned = self.structure_align_norm[idx](freq_nodes + freq_aligned)
            diff_aligned = self.structure_align_norm[idx](diff_nodes + diff_aligned)

            aligned_freq.append(freq_aligned.permute(0, 2, 1))
            aligned_diff.append(diff_aligned.permute(0, 2, 1))

        return aligned_freq, aligned_diff

    def _forward_impl(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch_size, _, _ = x_enc.shape

        if not self.use_frequency_branch and not self.use_temporal_branch:
            raise ValueError(
                "At least one of frequency or temporal branch must remain enabled."
            )

        multi_res_data = self.multi_res_data(x_enc)
        freq_enc = None
        temporal_enc = None

        if self.use_frequency_branch:
            freq_tokens = self.freq_embedding(multi_res_data)
            freq_enc = self.channel_resolution_encoder(freq_tokens, x_enc)

        if self.use_temporal_branch:
            temporal_tokens = self._build_temporal_tokens(multi_res_data)
            temporal_enc = self.channel_resolution_encoder(temporal_tokens, x_enc)

        if self.single_gnn:
            if self.use_frequency_branch and self.use_temporal_branch:
                fused_enc = self._fuse_branch_tokens(freq_enc, temporal_enc)
            elif self.use_frequency_branch:
                fused_enc = freq_enc
            else:
                fused_enc = temporal_enc
            graph_list, _ = self.shared_mrgnn.forward_features(fused_enc)
            output = torch.mean(torch.stack(graph_list, dim=-1), dim=-1)
            representation = output.reshape(batch_size, -1)
            logits = self.projection(representation)
            return logits, representation

        if self.use_frequency_branch and not self.use_temporal_branch:
            freq_graph_list, _ = self.freq_mrgnn.forward_features(freq_enc)
            output = torch.mean(torch.stack(freq_graph_list, dim=-1), dim=-1)
            representation = output.reshape(batch_size, -1)
            logits = self.projection(representation)
            return logits, representation

        if self.use_temporal_branch and not self.use_frequency_branch:
            diff_graph_list, _ = self.diff_mrgnn.forward_features(temporal_enc)
            output = torch.mean(torch.stack(diff_graph_list, dim=-1), dim=-1)
            representation = output.reshape(batch_size, -1)
            logits = self.projection(representation)
            return logits, representation

        freq_graph_list, freq_adj_list = self.freq_mrgnn.forward_features(freq_enc)
        diff_graph_list, diff_adj_list = self.diff_mrgnn.forward_features(temporal_enc)
        if self.use_structure_alignment:
            freq_graph_list, diff_graph_list = self._apply_structure_alignment(
                freq_graph_list, diff_graph_list, freq_adj_list, diff_adj_list
            )
        if self.use_cross_graph_interaction:
            freq_graph_list, diff_graph_list = self._cross_graph_interaction(
                freq_graph_list, diff_graph_list
            )

        freq_graph_out = torch.mean(torch.stack(freq_graph_list, dim=-1), dim=-1)
        diff_graph_out = torch.mean(torch.stack(diff_graph_list, dim=-1), dim=-1)
        freq_summary = freq_graph_out.mean(dim=-1)
        diff_summary = diff_graph_out.mean(dim=-1)
        graph_gate = torch.sigmoid(
            self.graph_fusion_gate(torch.cat([freq_summary, diff_summary], dim=-1))
        ).unsqueeze(-1)
        output = graph_gate * freq_graph_out + (1.0 - graph_gate) * diff_graph_out

        representation = output.reshape(batch_size, -1)
        logits = self.projection(representation)
        return logits, representation

    def forward_representation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        _, representation = self._forward_impl(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask
        )
        return representation

    def forward_branch_representations(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None
    ):
        if not self.use_frequency_branch or not self.use_temporal_branch:
            raise ValueError(
                "Both frequency and temporal branches must be enabled to extract branch representations."
            )

        multi_res_data = self.multi_res_data(x_enc)
        freq_tokens = self.freq_embedding(multi_res_data)
        temporal_tokens = self._build_temporal_tokens(multi_res_data)
        freq_enc = self.channel_resolution_encoder(freq_tokens, x_enc)
        temporal_enc = self.channel_resolution_encoder(temporal_tokens, x_enc)

        freq_per_res = torch.stack([token.mean(dim=1) for token in freq_enc], dim=1)
        temporal_per_res = torch.stack(
            [token.mean(dim=1) for token in temporal_enc], dim=1
        )
        freq_global = freq_per_res.mean(dim=1)
        temporal_global = temporal_per_res.mean(dim=1)

        return {
            "frequency_per_res": freq_per_res,
            "temporal_per_res": temporal_per_res,
            "frequency_global": freq_global,
            "temporal_global": temporal_global,
        }

    def forward_encoded_tokens(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        multi_res_data = self.multi_res_data(x_enc)
        encoded = {}

        if self.use_frequency_branch:
            freq_tokens = self.freq_embedding(multi_res_data)
            encoded["frequency"] = self.channel_resolution_encoder(freq_tokens, x_enc)

        if self.use_temporal_branch:
            temporal_tokens = self._build_temporal_tokens(multi_res_data)
            encoded["temporal"] = self.channel_resolution_encoder(
                temporal_tokens, x_enc
            )

        if not encoded:
            raise ValueError(
                "At least one of frequency or temporal branch must remain enabled."
            )
        return encoded

    def forward_graph_adjacencies(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.single_gnn:
            raise ValueError(
                "Adjacency visualization currently expects dual-graph mode, but single_gnn is enabled."
            )

        multi_res_data = self.multi_res_data(x_enc)
        adjacency_outputs = {}

        if self.use_frequency_branch:
            freq_tokens = self.freq_embedding(multi_res_data)
            freq_enc = self.channel_resolution_encoder(freq_tokens, x_enc)
            _, freq_adj_list = self.freq_mrgnn.forward_features(freq_enc)
            adjacency_outputs["frequency"] = freq_adj_list

        if self.use_temporal_branch:
            temporal_tokens = self._build_temporal_tokens(multi_res_data)
            temporal_enc = self.channel_resolution_encoder(temporal_tokens, x_enc)
            _, temporal_adj_list = self.diff_mrgnn.forward_features(temporal_enc)
            adjacency_outputs["temporal"] = temporal_adj_list

        if not adjacency_outputs:
            raise ValueError(
                "At least one of frequency or temporal branch must remain enabled."
            )

        return adjacency_outputs

    def _forward_with_adjs(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass that also returns adjacency matrices for structure loss computation."""
        batch_size, _, _ = x_enc.shape

        multi_res_data = self.multi_res_data(x_enc)
        freq_enc = None
        temporal_enc = None

        if self.use_frequency_branch:
            freq_tokens = self.freq_embedding(multi_res_data)
            freq_enc = self.channel_resolution_encoder(freq_tokens, x_enc)

        if self.use_temporal_branch:
            temporal_tokens = self._build_temporal_tokens(multi_res_data)
            temporal_enc = self.channel_resolution_encoder(temporal_tokens, x_enc)

        all_adj_list = []

        if self.single_gnn:
            if self.use_frequency_branch and self.use_temporal_branch:
                fused_enc = self._fuse_branch_tokens(freq_enc, temporal_enc)
            elif self.use_frequency_branch:
                fused_enc = freq_enc
            else:
                fused_enc = temporal_enc
            graph_list, adj_list = self.shared_mrgnn.forward_features(fused_enc)
            all_adj_list.extend(adj_list)
            output = torch.mean(torch.stack(graph_list, dim=-1), dim=-1)
            representation = output.reshape(batch_size, -1)
            logits = self.projection(representation)
            return logits, representation, all_adj_list

        if self.use_frequency_branch and not self.use_temporal_branch:
            freq_graph_list, freq_adj_list = self.freq_mrgnn.forward_features(freq_enc)
            all_adj_list.extend(freq_adj_list)
            output = torch.mean(torch.stack(freq_graph_list, dim=-1), dim=-1)
            representation = output.reshape(batch_size, -1)
            logits = self.projection(representation)
            return logits, representation, all_adj_list

        if self.use_temporal_branch and not self.use_frequency_branch:
            diff_graph_list, diff_adj_list = self.diff_mrgnn.forward_features(temporal_enc)
            all_adj_list.extend(diff_adj_list)
            output = torch.mean(torch.stack(diff_graph_list, dim=-1), dim=-1)
            representation = output.reshape(batch_size, -1)
            logits = self.projection(representation)
            return logits, representation, all_adj_list

        freq_graph_list, freq_adj_list = self.freq_mrgnn.forward_features(freq_enc)
        diff_graph_list, diff_adj_list = self.diff_mrgnn.forward_features(temporal_enc)
        all_adj_list.extend(freq_adj_list)
        all_adj_list.extend(diff_adj_list)

        if self.use_structure_alignment:
            freq_graph_list, diff_graph_list = self._apply_structure_alignment(
                freq_graph_list, diff_graph_list, freq_adj_list, diff_adj_list
            )
        if self.use_cross_graph_interaction:
            freq_graph_list, diff_graph_list = self._cross_graph_interaction(
                freq_graph_list, diff_graph_list
            )

        freq_graph_out = torch.mean(torch.stack(freq_graph_list, dim=-1), dim=-1)
        diff_graph_out = torch.mean(torch.stack(diff_graph_list, dim=-1), dim=-1)
        freq_summary = freq_graph_out.mean(dim=-1)
        diff_summary = diff_graph_out.mean(dim=-1)
        graph_gate = torch.sigmoid(
            self.graph_fusion_gate(torch.cat([freq_summary, diff_summary], dim=-1))
        ).unsqueeze(-1)
        output = graph_gate * freq_graph_out + (1.0 - graph_gate) * diff_graph_out

        representation = output.reshape(batch_size, -1)
        logits = self.projection(representation)
        return logits, representation, all_adj_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        logits, _ = self._forward_impl(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask
        )
        return logits

    def forward_with_adjs(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass returning logits, representation, and adjacency matrices."""
        return self._forward_with_adjs(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
