import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import weight_norm

from layers.Augmentation import get_augmentation


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # temp = self.pe[:, :, : x.size(2), :]
        return self.pe[:, :, : x.size(2), :]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)  # c_in is seq_length here
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_length)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)  # (batch_size, enc_in, d_model)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class CrossChannelTokenEmbedding(nn.Module):
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class ListPatchEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        d_model,
        patch_len_list,
        stride_list,
        dropout,
        augmentation=["none"],
        single_channel=False,
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]
        self.single_channel = single_channel

        linear_layers = [
            CrossChannelTokenEmbedding(
                c_in=enc_in if not single_channel else 1,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        self.value_embeddings = nn.ModuleList(linear_layers)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

        self.learnable_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in patch_len_list]
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)
        if self.single_channel:
            B, C, L = x.shape
            x = torch.reshape(x, (B * C, 1, L))

        x_list = []
        for padding, value_embedding in zip(self.paddings, self.value_embeddings):
            x_new = padding(x).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x_new = value_embedding(x_new)  # (batch_size, d_model, patch_num, 1)
            x_new = x_new.squeeze().transpose(1, 2)  # (batch_size, patch_num, d_model)
            # Per patch augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new = self.augmentation[aug_idx](x_new)
            x_list.append(x_new)

        x = [
            x + cxt + self.position_embedding(x)
            for x, cxt in zip(x_list, self.learnable_embeddings)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        return x


class Multi_Resolution_Data(nn.Module):
    """
    多分辨率数据处理模块
    通过不同大小的卷积核对时间序列数据进行下采样，生成多个不同分辨率的表示
    """
    def __init__(self, enc_in, resolution_list, stride_list):
        """
        初始化多分辨率数据处理模块 
        Args:
            enc_in (int): 输入特征维度（变量数量）
            resolution_list (list): 分辨率列表，每个元素定义一个卷积核大小，用于控制下采样比例
            stride_list (list): 步长列表，每个元素对应一个填充大小，用于处理边界情况
        """
        super(Multi_Resolution_Data, self).__init__()
        
        # 创建一组复制填充层，用于处理不同分辨率下的边界情况
        # 每个填充层对应一个特定的步长
        self.paddings = nn.ModuleList([nn.ReplicationPad1d((0, stride)) for stride in stride_list])

        # 创建一组1D卷积层，每个卷积层使用不同的卷积核大小和步长
        # 实现对输入数据的不同粒度下采样，生成多尺度表示
        self.multi_res = nn.ModuleList([
            nn.Conv1d(
                in_channels=enc_in,      # 输入通道数等于特征维度
                out_channels=enc_in,     # 输出通道数等于输入通道数，保持特征维度不变
                kernel_size=res,         # 卷积核大小等于分辨率值
                stride=res,              # 步长等于卷积核大小，实现下采样
                padding=0,               # 无额外填充
                padding_mode='circular'  # 循环填充模式
            )
            for res in resolution_list
        ])

    def forward(self, x):
        """
        前向传播函数
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, enc_in)
            
        Returns:
            list: 包含所有分辨率处理结果的列表，每个元素形状为 (batch_size, enc_in, new_seq_len)
        """
        # 调整输入张量的维度顺序，从 (batch_size, seq_len, enc_in) 转换为 (batch_size, enc_in, seq_len)
        x = x.permute(0, 2, 1)
        
        # 存储不同分辨率处理结果的列表
        x_list = []
        
        # 对每个分辨率设置进行处理
        for l in range(len(self.multi_res)):
            # 应用对应的填充层处理边界
            out = self.paddings[l](x)
            # 通过对应的卷积层进行下采样
            out = self.multi_res[l](out)
            # 将处理结果添加到输出列表中
            x_list.append(out)
            
        return x_list

class Frequency_Embedding(nn.Module):
    def __init__(self, d_model, res_len, augmentation=["none"]):
        super(Frequency_Embedding, self).__init__()
        # 保存模型维度参数
        self.d_model = d_model
        
        # 为每个分辨率创建一个线性变换层，用于频率嵌入
        # 输入维度为 res/2+1 (实数FFT的长度)，输出维度为 d_model/2+1
        # 转换为复数类型以处理频率数据
        self.embeddings = nn.ModuleList([
            nn.Linear(int(res/2)+1, int(self.d_model/2)+1).to(torch.cfloat)
            for res in res_len
        ])

        # 创建数据增强模块列表，支持多种增强方式
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

    def forward(self, x_list):
        # 存储处理后的输出结果
        x_out = []
        
        # 遍历每个输入张量
        for l in range(len(x_list)):
            # 对输入进行快速傅里叶变换(RFFT)，转换到频率域
            x = torch.fft.rfft(x_list[l], dim=-1)
            
            # 使用对应的嵌入层对频率特征进行线性变换
            out = self.embeddings[l](x)
            
            # 进行逆傅里叶变换(IRFFT)，转换回时域，n参数指定输出长度
            out = torch.fft.irfft(out, dim=-1, n=self.d_model)

            # 随机选择一种数据增强方式并应用
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            out = self.augmentation[aug_idx](out)
            
            # 将处理结果添加到输出列表
            x_out.append(out)

        # 返回处理后的结果列表
        return x_out


class SpectralExpertBranch(nn.Module):
    def __init__(self, freq_bins):
        super(SpectralExpertBranch, self).__init__()
        self.global_real = nn.Parameter(torch.zeros(1, 1, freq_bins))
        self.global_imag = nn.Parameter(torch.zeros(1, 1, freq_bins))
        self.local_mixer = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(8, 2, kernel_size=1),
        )

    def forward(self, x_fft):
        batch_size, channels, freq_bins = x_fft.shape
        global_filter = torch.complex(self.global_real, self.global_imag)
        global_out = x_fft * (1.0 + torch.tanh(global_filter))

        local_in = torch.stack([x_fft.real, x_fft.imag], dim=2)
        local_in = local_in.reshape(batch_size * channels, 2, freq_bins)
        local_out = self.local_mixer(local_in).reshape(batch_size, channels, 2, freq_bins)
        local_out = torch.complex(local_out[:, :, 0, :], local_out[:, :, 1, :])
        return global_out + local_out


class MultiBandFrequencyBlock(nn.Module):
    def __init__(self, seq_len, d_model, low_freq_ratio=0.5, dropout=0.1, augmentation=("none",)):
        super(MultiBandFrequencyBlock, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.freq_bins = seq_len // 2 + 1

        low_bins = max(1, min(self.freq_bins - 1, int(round(self.freq_bins * low_freq_ratio))))
        low_mask = torch.zeros(1, 1, self.freq_bins)
        low_mask[..., :low_bins] = 1.0
        high_mask = 1.0 - low_mask
        self.register_buffer("low_mask", low_mask, persistent=False)
        self.register_buffer("high_mask", high_mask, persistent=False)

        hidden_dim = max(self.freq_bins // 2, 8)
        self.pre_selector = nn.Sequential(
            nn.Linear(self.freq_bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.freq_bins),
        )
        self.band_gate = nn.Sequential(
            nn.Linear(self.freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.global_branch = SpectralExpertBranch(self.freq_bins)
        self.low_branch = SpectralExpertBranch(self.freq_bins)
        self.high_branch = SpectralExpertBranch(self.freq_bins)

        self.base_projection = nn.Linear(seq_len, d_model)
        self.spectral_projection = nn.Linear(seq_len, d_model)
        self.output_gate = nn.Linear(d_model * 2, d_model)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

        nn.init.constant_(self.output_gate.bias, -1.0)

    def _pre_filter(self, x_fft):
        power = torch.log1p(torch.abs(x_fft).pow(2).mean(dim=1))
        selector = torch.sigmoid(self.pre_selector(power)).unsqueeze(1)
        filtered_fft = x_fft * (0.5 + 0.5 * selector)
        return filtered_fft, power

    def forward(self, x):
        seq_len = x.shape[-1]
        window = torch.hann_window(seq_len, device=x.device, dtype=x.dtype).view(1, 1, seq_len)
        x_fft = torch.fft.rfft(x * window, dim=-1)

        filtered_fft, power = self._pre_filter(x_fft)
        gates = torch.softmax(self.band_gate(power), dim=-1)

        full_out = self.global_branch(filtered_fft)
        low_out = self.low_branch(filtered_fft * self.low_mask)
        high_out = self.high_branch(filtered_fft * self.high_mask)

        mixed_fft = (
            gates[:, 0].view(-1, 1, 1) * full_out
            + gates[:, 1].view(-1, 1, 1) * low_out
            + gates[:, 2].view(-1, 1, 1) * high_out
        )

        fused_fft = x_fft + torch.tanh(self.residual_scale) * mixed_fft
        spectral_seq = torch.fft.irfft(fused_fft, n=self.seq_len, dim=-1)
        spectral_tokens = self.spectral_projection(spectral_seq)
        base_tokens = self.base_projection(x)

        gate = torch.sigmoid(self.output_gate(torch.cat([base_tokens, spectral_tokens], dim=-1)))
        out = base_tokens + gate * (spectral_tokens - base_tokens)

        if self.training and len(self.augmentation) > 0:
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            out = self.augmentation[aug_idx](out)

        out = self.norm(out)
        return self.dropout(out)


class Frequency_Embedding(nn.Module):
    def __init__(
        self,
        d_model,
        res_len,
        augmentation=("none",),
        dropout=0.1,
        low_freq_ratio=0.5,
    ):
        super(Frequency_Embedding, self).__init__()
        self.blocks = nn.ModuleList(
            [
                MultiBandFrequencyBlock(
                    seq_len=seq_len,
                    d_model=d_model,
                    low_freq_ratio=low_freq_ratio,
                    dropout=dropout,
                    augmentation=augmentation,
                )
                for seq_len in res_len
            ]
        )

    def forward(self, x_list):
        return [block(x) for block, x in zip(self.blocks, x_list)]


class ChannelBranchEmbedding(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout=0.1, augmentation=("none",)):
        super(ChannelBranchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model

        # 多尺度卷积特征提取：简化为2个分支（小/大尺度），降低复杂度
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(enc_in, d_model // 2, kernel_size=k, padding=k // 2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1)
            )
            for k in [7, 31]  # 小/大感受野：局部细节+长周期趋势
        ])
        # 简化为平均融合，去除可学习参数
        self.scale_fusion = nn.Identity()

        self.channel_bias = nn.Parameter(torch.zeros(1, enc_in, seq_len))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

    def forward(self, x):
        # x: (B, seq_len, enc_in)
        x = x.transpose(1, 2)  # (B, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x = self.augmentation[aug_idx](x)
        x = x + self.channel_bias

        # 多尺度卷积特征提取：2分支简化版
        scale_feats = [conv(x) for conv in self.scale_convs]  # [(B, d_model//2, 1), ...]
        scale_feats = [f.squeeze(-1) for f in scale_feats]  # [(B, d_model//2), ...]
        # 简化为平均融合（降低参数量，减少过拟合风险）
        x = torch.stack(scale_feats, dim=-1).mean(dim=-1)  # (B, d_model//2)
        # 重复扩展到d_model（保持输出维度一致）
        x = x.repeat(1, 2)  # (B, d_model)

        # 扩展到每个通道
        x = x.unsqueeze(1).expand(-1, self.enc_in, -1)  # (B, enc_in, d_model)

        x = self.norm(x)
        return self.dropout(x)
