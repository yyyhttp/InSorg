# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, Linear
from mmengine.model import ModuleList
from torch import Tensor
import torch
from torch import nn
from mmdet.registry import MODELS
from mmdet.utils import MultiConfig
from .fcn_mask_head import FCNMaskHead
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class Head17CoarseMaskHead(FCNMaskHead):
    """Coarse mask head used in PointRend.

    Compared with standard ``FCNMaskHead``, ``CoarseMaskHead`` will downsample
    the input feature map instead of upsample it.

    Args:
        num_convs (int): Number of conv layers in the head. Defaults to 0.
        num_fcs (int): Number of fc layers in the head. Defaults to 2.
        fc_out_channels (int): Number of output channels of fc layer.
            Defaults to 1024.
        downsample_factor (int): The factor that feature map is downsampled by.
            Defaults to 2.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_convs: int = 0,
                 num_fcs: int = 2,
                 fc_out_channels: int = 1024,
                 downsample_factor: int = 2,
                 init_cfg: MultiConfig = dict(
                     type='Xavier',
                     override=[
                         dict(name='fcs'),
                         dict(type='Constant', val=0.001, name='fc_logits')
                     ]),
                 *arg,
                 **kwarg) -> None:
        super().__init__(
            *arg,
            num_convs=num_convs,
            upsample_cfg=dict(type=None),
            init_cfg=None,
            **kwarg)
        self.init_cfg = init_cfg
        self.num_fcs = num_fcs
        assert self.num_fcs > 0
        self.fc_out_channels = fc_out_channels
        self.downsample_factor = downsample_factor
        assert self.downsample_factor >= 1
        # remove conv_logit
        delattr(self, 'conv_logits')

        if downsample_factor > 1:
            downsample_in_channels = (
                self.conv_out_channels
                if self.num_convs > 0 else self.in_channels)
            self.downsample_conv = ConvModule(
                downsample_in_channels,
                self.conv_out_channels,
                kernel_size=downsample_factor,
                stride=downsample_factor,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        else:
            self.downsample_conv = None

        self.output_size = (self.roi_feat_size[0] // downsample_factor,
                            self.roi_feat_size[1] // downsample_factor)
        self.output_area = self.output_size[0] * self.output_size[1]

        last_layer_dim = self.conv_out_channels * self.output_area

        self.fcs = ModuleList()
        for i in range(num_fcs):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.fcs.append(Linear(fc_in_channels, self.fc_out_channels))
        last_layer_dim = self.fc_out_channels
        output_channels = self.num_classes * self.output_area
        self.fc_logits = Linear(last_layer_dim, output_channels)
        self.mynet=MyNetwork(in_channels=256)
        self.cbam=CBAM(in_planes=256)
        

    def init_weights(self) -> None:
        """Initialize weights."""
        super(FCNMaskHead, self).init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Extract mask RoI features.

        Returns:
            Tensor: Predicted foreground masks.
        """
        for conv in self.convs:
            x = conv(x)
        x = self.mynet(x)
        x= self.cbam(x)
        if self.downsample_conv is not None:
            x = self.downsample_conv(x)
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_preds = self.fc_logits(x).view(
            x.size(0), self.num_classes, *self.output_size)
        return mask_preds




import torch
import torch.nn as nn
import torch.fft


# 定义 FFT 和 IFFT 模块
class FFTModule(nn.Module):
    def __init__(self, channels):
        super(FFTModule, self).__init__()
        self.channels = channels

    def forward(self, x):
        # 对每个通道进行 2D FFT
        x_fft = torch.fft.fft2(x)
        return x_fft


class IFFTModule(nn.Module):
    def __init__(self, channels):
        super(IFFTModule, self).__init__()
        self.channels = channels

    def forward(self, x):
        # 对每个通道进行 2D IFFT
        x_ifft = torch.fft.ifft2(x)
        # 取实部，忽略虚部
        return torch.real(x_ifft)


# 定义整体网络结构
class MyNetwork(nn.Module):
    def __init__(self, in_channels=256):
        super(MyNetwork, self).__init__()
        # 输入通道为 256，分成两部分，每部分 128 通道
        self.in_channels = in_channels
        self.split_channels = in_channels // 2

        # FFT 和 IFFT 部分
        self.fft_module = FFTModule(channels=self.split_channels)
        self.ifft_module = IFFTModule(channels=self.split_channels)

        # 卷积部分
        self.conv1_fft = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1)
        self.conv2_fft = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1)

        # 常规卷积部分（另一个分支）
        self.conv1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 将输入的256通道分成两部分
        x1, x2 = torch.split(x, self.split_channels, dim=1)

        # 第一部分：FFT -> IFFT -> Conv3x3 -> Conv3x3
        x1_fft = self.fft_module(x1)
        x1_ifft = self.ifft_module(x1_fft)
        x1_ifft = torch.relu(self.conv1_fft(x1_ifft))
        x1_ifft = torch.relu(self.conv2_fft(x1_ifft))

        # 第二部分：Conv3x3 -> Conv3x3
        x2 = torch.relu(self.conv1(x2))
        x2 = torch.relu(self.conv2(x2))

        # 两部分的输出相加，形成最终的256通道输出
        out = torch.cat([x1_ifft, x2], dim=1)

        return out







import torch
from torch import nn

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图
