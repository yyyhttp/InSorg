# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.structures.bbox import (cat_boxes, empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)
from mmdet.utils import InstanceList, MultiConfig, OptInstanceList
from .anchor_head import AnchorHead
import torch
import torch.nn as nn
import torch.fft

@MODELS.register_module()
class DFPNHead(AnchorHead):
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 **kwargs) -> None:
        self.num_convs = num_convs

        assert num_classes == 1
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)
        self.dfpn =DfpnModule(dim=256)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.rpn_conv(x)
        x = self.dfpn(x)
        x = F.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) \
            -> dict:
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)
        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)
    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]
        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results



class DfpnModule(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bn0 = nn.BatchNorm2d(dim)
        self.relu0 = nn.ReLU()
        self.conv0_0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.bn0_0 = nn.BatchNorm2d(dim)
        self.relu0_0 = nn.ReLU()
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 9), padding=(0, 4), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (9, 1), padding=(4, 0), groups=dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)
        self.relu4 = nn.ReLU()
        self.fft_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn_fft = nn.BatchNorm2d(dim)
        self.relu_fft = nn.ReLU()

    def forward(self, x):
        u = x.clone()  
        an = self.conv0(x) 
        an = self.bn0(an)
        an = self.relu0(an)
        an = self.conv0_0_1(an)
        an = self.conv0_0_2(an)
        an = self.bn0_0(an)
        an = self.relu0_0(an)
        an_0 = self.conv0_1(an)
        an_0 = self.conv0_2(an_0)
        an_0 = self.bn1(an_0)
        an_0 = self.relu1(an_0)
        an_1 = self.conv1_1(an)
        an_1 = self.conv1_2(an_1)
        an_1 = self.bn2(an_1)
        an_1 = self.relu2(an_1)
        an_2 = self.conv2_1(an)
        an_2 = self.conv2_2(an_2)
        an_2 = self.bn3(an_2)
        an_2 = self.relu3(an_2)
        an = an + an_0 + an_1 + an_2 
        x_fft = torch.fft.fft2(x) 
        x_fft_shifted = torch.fft.fftshift(x_fft)  
        x_fft_processed = torch.fft.ifft2(x_fft_shifted)  
        x_fft_real = x_fft_processed.real 
        S0 = self.fft_conv(x_fft_real)  
        S0 = self.bn_fft(S0)
        S0 = self.relu_fft(S0)
        an = an + S0
        an = self.conv3(an)
        an = self.bn4(an)
        an = self.relu4(an)
        return an * u 

