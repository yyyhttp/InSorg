# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, Linear
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class Head15CoarseMaskHead(FCNMaskHead):
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
                 num_convs: int = 0, ################### 这里是加一层卷积
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
        # self.combinedA=CombinedAttention(channel=256)
        self.improveatt=Improved3AttentionModule(dim=256)

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
            # x =self.improv2att(x)
            # x=self.wtConv2d(x)
        x=self.improveatt(x)
        # x =self.akconv(x)
        # x=self.psa(x)
        # x=self.stokenatt(x)
        # x=self.crisscrossa(x)
        # x=self.parnet(x)
        # x=self.enhanceparnet(x)
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

class Improved3AttentionModule(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        # 使用3x3核的卷积层，应用深度卷积
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bn0 = nn.BatchNorm2d(dim)
        self.relu0 = nn.ReLU()

        # 增加1x3和3x1卷积
        self.conv0_0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.bn0_0 = nn.BatchNorm2d(dim)
        self.relu0_0 = nn.ReLU()

        # 两组卷积层，分别使用1x5和5x1核，用于跨度不同的特征提取，均应用深度卷积
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU()

        # 另外两组卷积层，使用更大的核进行特征提取，分别为1x7和7x1，也是深度卷积
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.ReLU()

        # 使用最大尺寸的核进行特征提取，为1x9和9x1，深度卷积
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 9), padding=(0, 4), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (9, 1), padding=(4, 0), groups=dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu3 = nn.ReLU()

        # 最后一个1x1卷积层，用于整合上述所有特征提取的结果
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        u = x.clone()  # 克隆输入x，以便之后与注意力加权的特征进行相乘
        attn = self.conv0(x)  # 应用初始的3x3卷积
        attn = self.bn0(attn)
        attn = self.relu0(attn)

        # 增加1x3和3x1卷积
        attn = self.conv0_0_1(attn)
        attn = self.conv0_0_2(attn)
        attn = self.bn0_0(attn)
        attn = self.relu0_0(attn)

        # 应用1x5和5x1卷积，进一步提取特征
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.bn1(attn_0)
        attn_0 = self.relu1(attn_0)

        # 应用1x7和7x1卷积，进一步提取特征
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.bn2(attn_1)
        attn_1 = self.relu2(attn_1)

        # 应用1x9和9x1卷积，进一步提取特征
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.bn3(attn_2)
        attn_2 = self.relu3(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2  # 将所有特征提取的结果相加

        attn = self.conv3(attn)  # 应用最后的1x1卷积层整合特征
        attn = self.bn4(attn)
        attn = self.relu4(attn)

        return attn * u

