import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    """De-stationary Attention"""

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = (
            0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        )  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        causal=False,
        bucket_size=4,
        n_hashes=4,
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat(
                [queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1
            )

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(
        self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1
    ):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            FullAttention(
                False,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention,
            ),
            d_model,
            n_heads,
        )
        self.dim_sender = AttentionLayer(
            FullAttention(
                False,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention,
            ),
            d_model,
            n_heads,
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(
                False,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention,
            ),
            d_model,
            n_heads,
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")

        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(
            dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch
        )

        batch_router = repeat(
            self.router,
            "seg_num factor d_model -> (repeat seg_num) factor d_model",
            repeat=batch,
        )

        dim_buffer, attn = self.dim_sender(
            batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None
        )

        dim_receive, attn = self.dim_receiver(
            dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None
        )
        # dim_receive, attn = self.dim_receiver(dim_send, dim_send, dim_send, attn_mask=None, tau=None, delta=None)

        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)


        final_out = rearrange(
            dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch
        )

        return final_out


class MedformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
    ):
        super().__init__()

        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        if no_inter:
            self.inter_attention = None
        else:
            self.inter_attention = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))

        x_intra = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_intra.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)

        if self.inter_attention is not None:

            routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)  # (B, N, D)
            x_inter, attn_inter = self.inter_attention(
                routers, routers, routers, attn_mask=None, tau=tau, delta=delta
            )
            x_out = [
                torch.cat([x[:, :-1], x_inter[:, i : i + 1]], dim=1)  # (B, Li, D)
                for i, x in enumerate(x_intra)
            ]
            attn_out += [attn_inter]
        else:
            x_out = x_intra
        return x_out, attn_out

class FormerLayer(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super().__init__()

        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))

        x_out = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_out.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)

        return x_out, attn_out


class CoTAR(nn.Module):
    def __init__(self, d_model, d_core=None):
        super(CoTAR, self).__init__()
        d_core = d_core or max(d_model // 4, 1)
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_core)
        self.lin3 = nn.Linear(d_model + d_core, d_model)
        self.lin4 = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, token_num, dim = x.shape
        core = torch.nn.functional.gelu(self.lin1(x))
        core = self.lin2(core)
        weight = torch.softmax(core, dim=1)
        core = torch.sum(core * weight, dim=1, keepdim=True).repeat(1, token_num, 1)
        fused = torch.cat([x, core], dim=-1)
        fused = torch.nn.functional.gelu(self.lin3(fused))
        return self.lin4(fused)


class CentralizedMixerLayer(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(CentralizedMixerLayer, self).__init__()
        d_ff = d_ff or 2 * d_model
        self.mixer = CoTAR(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm1(x + self.mixer(x))
        x = self.norm2(x + self.mlp(x))
        return x


class ResolutionRouter(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResolutionRouter, self).__init__()
        self.router = CentralizedMixerLayer(d_model, d_ff=2 * d_model, dropout=dropout)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_list):
        if len(x_list) <= 1:
            return x_list, None

        pooled = torch.stack([x.mean(dim=1) for x in x_list], dim=1)
        routed = self.router(pooled)

        fused_list = []
        for idx, x in enumerate(x_list):
            context = routed[:, idx : idx + 1, :].expand(-1, x.size(1), -1)
            gate = torch.sigmoid(self.fusion(torch.cat([x, context], dim=-1)))
            fused_list.append(self.norm(x + gate * context))

        return fused_list, routed

class DifferenceAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DifferenceAttention, self).__init__()
        self.scale = scale                          # 缩放因子
        self.mask_flag = mask_flag                  # 是否使用掩码
        self.output_attention = output_attention    # 是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # Dropout层

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape    # B:批次大小, L:查询序列长度, H:头数, E:查询特征维度
        _, S, _, D = values.shape     # S:键/值序列长度, D:值特征维度
        
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention: softmax(Q*K^T/sqrt(d_k))
        
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            # 返回连续的输出张量和注意力权重
            return V.contiguous(), A
        else:
            # 只返回输出张量
            return V.contiguous(), None


class DifferenceAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(DifferenceAttentionLayer, self).__init__()

        # 计算键和值的维度，如果未指定则默认使用d_model/n_heads
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # 内部注意力机制实例(如DifferenceAttention)
        self.inner_attention = attention
        
        # 线性投影层：将输入特征映射到多头注意力所需的维度
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)      # 查询投影
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)        # 键投影
        self.value_projection = nn.Linear(d_model, d_values * n_heads)    # 值投影
        
        # 输出投影层：将多头注意力输出映射回模型维度
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        # 注意力头数量
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape    # B:批次大小, L:查询序列长度
        _, S, _ = keys.shape       # S:键/值序列长度
        H = self.n_heads           # 注意力头数

        queries = self.query_projection(queries).view(B, L, H, -1)  # (B, L, H, d_keys)
        keys = self.key_projection(keys).view(B, S, H, -1)          # (B, S, H, d_keys)
        values = self.value_projection(values).view(B, S, H, -1)    # (B, S, H, d_values)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)  # (B, L, H * d_values)

        return self.out_projection(out), attn

class DifferenceFormerlayer(nn.Module):
    def __init__(self, enc_in, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super(DifferenceFormerlayer, self).__init__()
        
        self.intra_attentions = nn.ModuleList(
            [
                DifferenceAttentionLayer(
                    DifferenceAttention(
                        False,                    # mask_flag: 不使用掩码
                        factor=1,                 # factor: 缩放因子
                        attention_dropout=dropout, # dropout比率
                        output_attention=output_attention, # 是否输出注意力权重
                    ),
                    d_model,   # 模型维度
                    n_heads,   # 注意力头数
                )
                for _ in range(num_blocks)  # 创建num_blocks个注意力层
            ]
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))

        x_out = []      # 存储每层的输出结果
        attn_out = []   # 存储每层的注意力权重(如果启用)
        
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            
            x_out.append(_x_out)    # (B, Li, D) - 每个输入对应的输出
            attn_out.append(_attn)  # 注意力权重

        return x_out, attn_out
