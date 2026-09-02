import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入自定义模块
from layers.Embed import Multi_Resolution_Data, Frequency_Embedding
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FormerLayer, DifferenceFormerlayer
from layers.Multi_Resolution_GNN import MRGNN
from layers.Difference_Pre import DifferenceDataEmb, DataRestoration

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.enc_in = configs.enc_in              # 输入特征维度(节点数)
        self.seq_len = configs.seq_len            # 输入序列长度
        self.d_model = configs.d_model            # 模型维度
        self.d_ff = configs.d_ff                  # 前馈网络维度
        self.n_heads = configs.n_heads            # 注意力头数
        self.e_layers = configs.e_layers          # 编码器层数
        self.dropout = configs.dropout            # dropout率
        self.output_attention = configs.output_attention  # 是否输出注意力权重
        self.activation = configs.activation      # 激活函数类型
        # 解析分辨率列表，转换为整数列表
        self.resolution_list = list(map(int, configs.resolution_list.split(",")))

        # 通道分割参数
        self.t = configs.t
        self.n = configs.n
        if configs.learnab:
            self.a = nn.Parameter(torch.tensor(configs.a))
            self.b = nn.Parameter(torch.tensor(configs.b))
        else:
            self.a = configs.a
            self.b = configs.b

        # 计算分辨率相关参数
        self.res_num = len(self.resolution_list)  # 分辨率数量
        self.stride_list = self.resolution_list   # 步长列表
        # 计算每种分辨率下的序列长度
        self.res_len = [int(self.seq_len//res)+1 for res in self.resolution_list]
        # 数据增强方式列表
        self.augmentations = configs.augmentations.split(",")

        # step1: 多分辨率数据生成
        # 创建多分辨率数据处理模块，用于生成不同时间分辨率的输入数据
        self.multi_res_data = Multi_Resolution_Data(self.enc_in, self.resolution_list, self.stride_list)

        # step2.1: 频域卷积网络
        # 频率嵌入模块，用于提取不同分辨率数据的频域特征
        self.freq_embedding = Frequency_Embedding(self.d_model, self.res_len, self.augmentations)

        # step2.2: 差分注意力网络
        # 差分数据嵌入模块，用于生成多分辨率数据的差分表示
        self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)
        # 差分注意力编码器，用于处理差分特征
        self.difference_attention = Encoder(
            [
                EncoderLayer(
                    DifferenceFormerlayer(
                        self.enc_in,           # 输入特征维度
                        self.res_num,          # 分辨率数量
                        self.d_model,          # 模型维度
                        self.n_heads,          # 注意力头数
                        self.dropout,          # dropout率
                        self.output_attention  # 是否输出注意力
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)  # 创建指定层数的编码器
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),  # 层归一化
        )
        # 数据恢复模块，用于将差分特征还原为原始表示
        self.data_restoration = DataRestoration(self.res_num, self.enc_in, self.d_model)
        # 为每种分辨率创建线性嵌入层
        self.embeddings = nn.ModuleList([nn.Linear(res_len, self.d_model) for res_len in self.res_len])

        # step 3: Transformer编码器
        # 主Transformer编码器，用于融合多分辨率特征
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FormerLayer(
                        len(self.resolution_list),  # 分辨率数量
                        configs.d_model,            # 模型维度
                        configs.n_heads,            # 注意力头数
                        configs.dropout,            # dropout率
                        configs.output_attention    # 是否输出注意力
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)  # 创建指定层数的编码器
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),  # 层归一化
        )

        # step 4: 多分辨率图神经网络
        # 多分辨率图神经网络模块，用于学习节点间的关系
        self.mrgnn = MRGNN(configs, self.res_len)

        # step 5: 输出投影
        # 线性投影层，将特征映射到分类结果
        self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 输入数据形状: B(批次大小), T(时间步), C(特征维度)
        B, T, C = x_enc.shape

        # 应用通道分割机制
        x_enc_new = x_enc.clone()
        if C >= 2 * self.n:  # 确保通道数足够分割
            front_8_half = x_enc[:, :, :self.n]
            back_8_half = x_enc[:, :, -self.n:]
            added_half = front_8_half * self.a + back_8_half * self.b

            if self.t > 0:
                x_enc_new[:, :, :self.n] = added_half
            else:
                x_enc_new[:, :, -self.n:] = added_half

        # step1: 生成多分辨率数据
        # 将输入数据转换为多种时间分辨率表示
        multi_res_data = self.multi_res_data(x_enc_new)

        # step2.1: 频域特征提取
        # 通过频域卷积网络提取各分辨率下的频域特征
        enc_out_1 = self.freq_embedding(multi_res_data)

        # step2.2: 差分注意力处理
        # 生成差分数据嵌入和填充掩码
        x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
        # 通过差分注意力网络处理差分特征
        x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)
        # 恢复差分处理后的数据
        enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
        # 通过线性层将各分辨率数据映射到统一的模型维度
        enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]

        # step 3: Transformer特征融合
        # 将频域特征和差分特征相加，作为Transformer输入
        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]
        # 通过主Transformer编码器进行特征融合
        enc_out, attns = self.encoder(data_enc, attn_mask=None)

        # step 4: 图神经网络处理
        # 通过多分辨率图神经网络学习节点间关系
        output, adjacency_matrix_list = self.mrgnn(enc_out)

        # step 5: 输出投影
        # 将图网络输出展平并投影到分类空间
        output = output.reshape(B, -1)  # 展平特征
        output = self.projection(output)  # 投影到分类结果

        return output  # 返回最终分类结果
