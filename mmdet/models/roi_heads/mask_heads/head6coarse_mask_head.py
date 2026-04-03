# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, Linear
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class Head6CoarseMaskHead(FCNMaskHead):
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




import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个函数生成负无穷大的矩阵，用于注意力机制中遮罩操作
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

# 定义一个交叉注意力模块
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels# 输入通道数
        self.channels = in_channels // 8# 缩减的通道数，为输入通道数的1/8
        # 定义三个1x1卷积层用于生成query、key和value
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)# 定义一个softmax层，用于计算注意力权重
        self.INF = INF # 引用之前定义的INF函数
        self.gamma = nn.Parameter(torch.zeros(1)) # 定义一个学习参数gamma，用于调节注意力的影响

    def forward(self, x):
        b, _, h, w = x.size()

        # 生成query
        query = self.ConvQuery(x)

        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)

        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # 生成key
        key = self.ConvKey(x)

        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)

        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # 生成value
        value = self.ConvValue(x)

        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)

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

        return self.gamma * (out_H + out_W) + x