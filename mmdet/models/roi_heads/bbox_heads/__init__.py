# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .multi_instance_bbox_head import MultiInstanceBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
#
from .SKAconvfc_bbox_head import (SKAConvFCBBoxHead, SKAShared2FCBBoxHead,
                               SKAShared4Conv1FCBBoxHead) #EMA检测头
from .PPMconvfc_bbox_head import (PPMConvFCBBoxHead, PPMShared2FCBBoxHead,
                               PPMShared4Conv1FCBBoxHead) #EMA检测头
from .EMAAconvfc_bbox_head import (EMAAConvFCBBoxHead, EMAAShared2FCBBoxHead,
                               EMAAShared4Conv1FCBBoxHead) #EMA检测头                               
from .PPAconvfc_bbox_head import (PPAConvFCBBoxHead, PPAShared2FCBBoxHead,
                               PPAShared4Conv1FCBBoxHead) #HFCET检测头
from .PPM2convfc_bbox_head import (PPM2ConvFCBBoxHead, PPM2Shared2FCBBoxHead,
                               PPM2Shared4Conv1FCBBoxHead) #zai share2 上的PPM的检测头
                               
from .PPM3convfc_bbox_head import (PPM3ConvFCBBoxHead, PPM3Shared2FCBBoxHead,
                               PPM3Shared4Conv1FCBBoxHead) #zai share2 上的PPM3的检测头
                               
__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'MultiInstanceBBoxHead','SKAShared2FCBBoxHead',
    'SKAShared4Conv1FCBBoxHead','SKAConvFCBBoxHead','PPMConvFCBBoxHead', 'PPMShared2FCBBoxHead',
                               'PPMShared4Conv1FCBBoxHead',
                               'EMAAShared2FCBBoxHead',
    'EMAAShared4Conv1FCBBoxHead','EMAAConvFCBBoxHead',
    'PPAConvFCBBoxHead', 'PPAShared2FCBBoxHead',
                               'PPAShared4Conv1FCBBoxHead',
                               'PPM2ConvFCBBoxHead', 'PPM2Shared2FCBBoxHead',
                               'PPM2Shared4Conv1FCBBoxHead', 'PPM3ConvFCBBoxHead', 'PPM3Shared2FCBBoxHead',
                               'PPM3Shared4Conv1FCBBoxHead',
    
]
