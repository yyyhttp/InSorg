# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, Linear
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class Head11CoarseMaskHead(FCNMaskHead):
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
        # self.crisscrossa =CrissCrossAttention(in_channels=256)
        # self.coord = CoordAtt(256, 256)
        self.enhanceparnet=EnhancedParNetAttention()

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
        # x=self.crisscrossa(x)
        # x=self.parnet(x)
        x=self.enhanceparnet(x)
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


import torch
import torch.nn as nn


class EnhancedParNetAttention(nn.Module):
    def __init__(self, channel=256):
        super().__init__()

        # Squeeze-and-Excitation (SE) 模块
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        # 1x1 卷积层和 3x3 卷积层，5x5 卷积层，通道数保持一致
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=5, padding=2),
            nn.BatchNorm2d(channel)
        )

        # 1x1 卷积层来调整通道数
        self.reduce_channels = nn.Conv2d(channel * 3, channel, kernel_size=1)

        # SiLU 激活函数
        self.silu = nn.SiLU()

        # 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 应用不同的卷积层
        x1 = self.conv1x1(x)  # 通道数为 channel
        x2 = self.conv3x3(x)  # 通道数为 channel
        x3 = self.conv5x5(x)  # 通道数为 channel

        # 应用 Squeeze-and-Excitation (SE) 模块
        se_weight = self.sse(x)
        x3 = se_weight * x

        # 连接特征并调整通道数
        x_cat = torch.cat((x1, x2, x3), dim=1)  # 拼接后的通道数是 3 * channel (即 768)

        # 使用卷积层调整通道数
        x_cat = self.reduce_channels(x_cat)  # 通道数调整为 channel (即 256)

        # 计算空间注意力图，注意力图的通道数为 1
        att_map = self.spatial_att(x_cat)

        # 扩展注意力图以匹配 x_cat 的通道数
        att_map = att_map.expand_as(x_cat)

        # 应用空间注意力机制
        y = self.silu(x_cat * att_map)

        return y
