import torch
import torch.nn as nn
#实现了多种时间序列数据增强方法，用于提升模型的鲁棒性和泛化能力
#包括随机噪声添加、缩放、翻转、通道 shuffle 等操作

class Jitter(nn.Module):
    """
    抖动增强：对每个元素添加高斯噪声
    通过在输入数据上添加随机噪声来模拟真实数据中的测量误差或环境干扰
    """
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale  # 噪声强度控制参数

    def forward(self, x):
        if self.training:  # 仅在训练模式下应用
            # 生成与输入x形状相同的随机噪声，并乘以缩放因子
            x += torch.randn_like(x) * self.scale
        return x


class Scale(nn.Module):
    """
    缩放增强：对每个通道乘以随机标量
    通过随机缩放不同通道的幅度来模拟传感器增益变化或信号强度波动
    """
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale  # 缩放强度控制参数

    def forward(self, x):
        if self.training:  # 仅在训练模式下应用
            B, C, T = x.shape  # B:批次大小, C:通道数, T:时间步长
            # 为每个通道生成随机缩放因子，形状为(B, C, 1)
            x *= 1 + torch.randn(B, C, 1, device=x.device) * self.scale
        return x


class Flip(nn.Module):
    """
    翻转增强：左右翻转时间序列
    通过时间轴反转来增强模型对时间顺序变化的鲁棒性
    """
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob  # 翻转概率

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            # 在时间维度(dim=-1)上进行翻转
            return torch.flip(x, [-1])
        return x


class Shuffle(nn.Module):
    """
    通道重排增强：随机打乱通道顺序
    通过重新排列不同传感器或特征的顺序来增强模型对通道顺序变化的鲁棒性
    """
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob  # 重排概率

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            B, C, T = x.shape  # B:批次大小, C:通道数, T:时间步长
            # 生成通道的随机排列索引
            perm = torch.randperm(C)
            # 应用通道重排
            return x[:, perm, :]
        return x


class TemporalMask(nn.Module):
    """
    时域掩码增强：随机屏蔽部分时间戳
    通过将某些时间步的数据置零来模拟传感器故障或数据丢失情况
    """
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio  # 掩码比例

    def forward(self, x):
        if self.training:
            B, C, T = x.shape  # B:批次大小, C:通道数, T:时间步长
            # 计算需要掩码的时间步数量
            num_mask = int(T * self.ratio)
            # 随机选择要掩码的时间步索引
            mask_indices = torch.randperm(T)[:num_mask]
            # 将选定时间步的所有通道数据置零
            x[:, :, mask_indices] = 0
        return x


class FrequencyMask(nn.Module):
    """
    频域掩码增强：在频域中随机屏蔽部分频率成分
    通过在频域中应用掩码来模拟频率选择性干扰或滤波器效应
    """
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio  # 频域掩码比例

    def forward(self, x):
        if self.training:
            B, C, T = x.shape  # B:批次大小, C:通道数, T:时间步长
            
            # 执行实数快速傅里叶变换(RFFT)，转换到频域
            x_fft = torch.fft.rfft(x, dim=-1)
            
            # 生成随机掩码矩阵，大于阈值的频率成分保留
            mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
            
            # 应用频域掩码
            x_fft = x_fft * mask
            
            # 执行逆实数快速傅里叶变换(IRFFT)，转换回时域
            # n=T参数确保输出长度与原始输入相同
            x = torch.fft.irfft(x_fft, n=T, dim=-1)
        return x


def get_augmentation(augmentation):
    """
    增强方法工厂函数：根据字符串参数创建对应的数据增强模块
    
    参数:
        augmentation (str): 增强方法名称和参数的字符串
        
    返回:
        nn.Module: 对应的数据增强模块
        
    支持的增强方法:
        - "jitter[scale]": 抖动增强，scale为可选参数
        - "scale[scale]": 缩放增强，scale为可选参数  
        - "drop[rate]": Dropout增强，rate为可选参数
        - "flip[prob]": 翻转增强，prob为可选参数
        - "shuffle[prob]": 通道重排增强，prob为可选参数
        - "frequency[ratio]": 频域掩码增强，ratio为可选参数
        - "mask[ratio]": 时域掩码增强，ratio为可选参数
        - "none": 无增强
    """
    if augmentation.startswith("jitter"):
        if len(augmentation) == 6:
            return Jitter()  # 默认scale=0.1
        return Jitter(float(augmentation[6:]))  # 解析scale参数
    elif augmentation.startswith("scale"):
        if len(augmentation) == 5:
            return Scale()  # 默认scale=0.1
        return Scale(float(augmentation[5:]))  # 解析scale参数
    elif augmentation.startswith("drop"):
        if len(augmentation) == 4:
            return nn.Dropout(0.1)  # 默认dropout=0.1
        return nn.Dropout(float(augmentation[4:]))  # 解析dropout参数
    elif augmentation.startswith("flip"):
        if len(augmentation) == 4:
            return Flip()  # 默认prob=0.5
        return Flip(float(augmentation[4:]))  # 解析prob参数
    elif augmentation.startswith("shuffle"):
        if len(augmentation) == 7:
            return Shuffle()  # 默认prob=0.5
        return Shuffle(float(augmentation[7:]))  # 解析prob参数
    elif augmentation.startswith("frequency"):
        if len(augmentation) == 9:
            return FrequencyMask()  # 默认ratio=0.1
        return FrequencyMask(float(augmentation[9:]))  # 解析ratio参数
    elif augmentation.startswith("mask"):
        if len(augmentation) == 4:
            return TemporalMask()  # 默认ratio=0.1
        return TemporalMask(float(augmentation[4:]))  # 解析ratio参数
    elif augmentation == "none":
        return nn.Identity()  # 无增强，返回恒等映射
    else:
        raise ValueError(f"Unknown augmentation {augmentation}")  # 未知增强方法