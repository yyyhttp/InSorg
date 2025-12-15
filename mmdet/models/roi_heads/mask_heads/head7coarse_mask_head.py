# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, Linear
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class Head7CoarseMaskHead(FCNMaskHead):
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
        # self.atten=AttentionModule(dim=256)
        # self.skconv=SKConv(in_ch=256)
        self.crisscrossa =CrissCrossAttention(in_channels=256)

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
        # x =self.akconv(x)
        # x=self.psa(x)
        # x=self.stokenatt(x)
        x=self.crisscrossa(x)
        # x=self.skconv(x)
        ###########
        if self.downsample_conv is not None:
            x = self.downsample_conv(x)

        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_preds = self.fc_logits(x).view(
            x.size(0), self.num_classes, *self.output_size)
        return mask_preds




#
#
# 引入多尺度特征：在网络中添加多个具有不同膨胀率（dilation rate）的卷积层，用以捕捉不同尺度的特征，这对于小目标的识别会更加有效。
# 融合多尺度特征：通过torch.stack将多个不同尺度的特征堆叠成一个张量，然后在第0维度上进行求和融合。
# 保持原有的结构和输入输出通道：确保改进后的模块在整体架构中可以无缝衔接，且输入输出通道保持一致。
# 通过上述改进，可以更好地捕捉小目标的细节，提高分割小目标的效果。

import torch
import torch.nn as nn

# 定义一个函数生成负无穷大的矩阵，用于注意力机制中遮罩操作
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

# 定义一个交叉注意力模块
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.channels = in_channels // 8  # 缩减的通道数，为输入通道数的1/8

        # 定义三个1x1卷积层用于生成query、key和value
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)  # 定义一个softmax层，用于计算注意力权重
        self.INF = INF  # 引用之前定义的INF函数
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义一个学习参数gamma，用于调节注意力的影响

        # 增加多尺度特征的卷积层
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=3, dilation=3)
        ])

    def forward(self, x):
        b, _, h, w = x.size()  # 获取输入的维度信息：批次大小、高度、宽度

        # 生成多尺度特征并融合
        multi_scale_features = [conv(x) for conv in self.multi_scale_conv]
        multi_scale_features = torch.stack(multi_scale_features)  # 将多尺度特征堆叠成一个张量
        multi_scale_features = torch.sum(multi_scale_features, dim=0)  # 在第0维度上进行求和，融合多尺度特征

        # 生成query、key、value
        query = self.ConvQuery(multi_scale_features)  # 通过1x1卷积生成query特征图
        key = self.ConvKey(multi_scale_features)  # 通过1x1卷积生成key特征图
        value = self.ConvValue(multi_scale_features)  # 通过1x1卷积生成value特征图

        # 处理query
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # 对query进行维度变换
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # 处理key
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # 对key进行维度变换
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # 处理value
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # 对value进行维度变换
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # 计算水平和垂直方向的注意力分数，并应用负无穷大遮罩
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)

        # 合并水平和垂直方向的注意力分数，并通过softmax归一化
        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))

        # 分离水平和垂直方向的注意力，并应用到value上
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        # 根据注意力分数加权value，并将水平和垂直方向的结果相加
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        # 返回加权结果并通过gamma参数调整其影响，同时加上输入x实现残差连接
        return self.gamma * (out_H + out_W) + x  # 保持输入输出通道256，针对小目标分割进行改进

# 由于对INF函数不了解，这里假设INF是一个已经定义的函数。
