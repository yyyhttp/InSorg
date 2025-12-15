# Copyright (c) OpenMMLab. All rights reserved.
from .coarse_mask_head import CoarseMaskHead
from .dynamic_mask_head import DynamicMaskHead
from .fcn_mask_head import FCNMaskHead
from .feature_relay_head import FeatureRelayHead
from .fused_semantic_head import FusedSemanticHead
from .global_context_head import GlobalContextHead
from .grid_head import GridHead
from .htc_mask_head import HTCMaskHead
from .mask_point_head import MaskPointHead
from .maskiou_head import MaskIoUHead
from .scnet_mask_head import SCNetMaskHead
from .scnet_semantic_head import SCNetSemanticHead
from .Head1coarse_mask_head import Head1CoarseMaskHead   
from .head2coarse_mask_head import Head2CoarseMaskHead ##
from .head3coarse_mask_head import Head3CoarseMaskHead ##
from .head4coarse_mask_head import Head4CoarseMaskHead ##
from .head5coarse_mask_head import Head5CoarseMaskHead ##
from .head6coarse_mask_head import Head6CoarseMaskHead ##
from .head7coarse_mask_head import Head7CoarseMaskHead ##
from .head8coarse_mask_head import Head8CoarseMaskHead ##
from .head9coarse_mask_head import Head9CoarseMaskHead ## HCFNET
from .head10coarse_mask_head import Head10CoarseMaskHead
from .head11coarse_mask_head import Head11CoarseMaskHead
from .head12coarse_mask_head import Head12CoarseMaskHead
from .head13coarse_mask_head import Head13CoarseMaskHead
from .head14coarse_mask_head import Head14CoarseMaskHead
from .head15coarse_mask_head import Head15CoarseMaskHead
from .head16coarse_mask_head import Head16CoarseMaskHead
from .head17coarse_mask_head import Head17CoarseMaskHead
from .head18coarse_mask_head import Head18CoarseMaskHead
from .head20coarse_mask_head import Head20CoarseMaskHead 
__all__ = [
    'FCNMaskHead', 'HTCMaskHead', 'FusedSemanticHead', 'GridHead',
    'MaskIoUHead', 'CoarseMaskHead', 'MaskPointHead', 'SCNetMaskHead',
    'SCNetSemanticHead', 'GlobalContextHead', 'FeatureRelayHead',
    'DynamicMaskHead','Head1CoarseMaskHead','Head2CoarseMaskHead','Head3CoarseMaskHead', 'Head4CoarseMaskHead',
    'Head5CoarseMaskHead','Head6CoarseMaskHead','Head7CoarseMaskHead','Head8CoarseMaskHead','Head9CoarseMaskHead',
    'Head10CoarseMaskHead','Head11CoarseMaskHead','Head12CoarseMaskHead','Head13CoarseMaskHead','Head14CoarseMaskHead','Head15CoarseMaskHead',
     'Head16CoarseMaskHead','Head17CoarseMaskHead','Head18CoarseMaskHead','Head20CoarseMaskHead',
    
]
 