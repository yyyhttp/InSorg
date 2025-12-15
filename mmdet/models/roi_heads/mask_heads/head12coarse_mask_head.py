# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, Linear
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class Head12CoarseMaskHead(FCNMaskHead):
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
        self.combinedA=CombinedAttention(channel=256)

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
        # x=self.enhanceparnet(x)
        # x=self.skconv(x)
        x=self.combinedA(x)
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

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CombinedAttention(nn.Module):
    def __init__(self, channel=256):
        super(CombinedAttention, self).__init__()
        self.channel = channel
        self.channels = channel // 8

        self.ConvQuery = nn.Conv2d(self.channel, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.channel, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.channel, self.channel, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=3, dilation=3)
        ])

        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channel, self.channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1),
            nn.BatchNorm2d(self.channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel)
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        b, _, h, w = x.size()

        # Criss-Cross Attention
        multi_scale_features = [conv(x) for conv in self.multi_scale_conv]
        multi_scale_features = torch.stack(multi_scale_features)
        multi_scale_features = torch.sum(multi_scale_features, dim=0)

        query = self.ConvQuery(multi_scale_features)
        key = self.ConvKey(multi_scale_features)
        value = self.ConvValue(multi_scale_features)

        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)

        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))

        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        x = self.gamma * (out_H + out_W) + x

        # ParNet Attention
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)

        return y
