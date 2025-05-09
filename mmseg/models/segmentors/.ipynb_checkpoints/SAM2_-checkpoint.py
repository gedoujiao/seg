# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Dict, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

##add hdd 2025/2/28
from project.sam_inference_demo import sam2
##add hdd 2025/3/3
from mmengine.runner import load_checkpoint



model_zoo = {
    'large':
    '/root/autodl-tmp/sam2_hiera_large.pt',
    'base':
    '/root/autodl-tmp/sam2_hiera_large.pt',
    'small':
    '',
    'tiny':
    '',
}

@MODELS.register_module()
class SAM2PromptSTAdv(BaseSegmentor):
    def __init__(self,
                 backbone: ConfigType,
                 pretrained: Optional[str] = None,
                 SAM2_config:OptConfigType = None,
                 SAM_arch: str = 'large',
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None
                ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = self.init_SAM2_model(SAM2_config,SAM_arch)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head
        
    #add hdd 2025/2/28
    def init_SAM2_model(self, cfg: str, arch: str):
        model = MODELS.build(cfg)
        load_checkpoint(model, model_zoo.get(arch), strict=True)
        return model

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad= requires_grad
    
    #add hdd 2025/2/28
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        
        ##data_preprocessor
        data = self.data_preprocessor(data, True)
        
        ##step0:frozen 
        self.set_requires_grad(self.backbone, False)
        self.set_requires_grad(self.neck, False)
        self.set_requires_grad(self.decode_head, False)

        ##step1:optimize segmentor
        self.set_requires_grad(self.neck, True)
        self.set_requires_grad(self.decode_head, True)
        ## 1.0 forward SAM segmentor
        
        #SAM_backbone_out = self.SAM_backbone_forward(data['inputs'], data['B_inputs'])
        SAM_backbone_out = self.backbone.module.image_encoder(data['inputs'])
        #
        all_out1 = self.neck.module(SAM_backbone_out['fb_s'])
        all_out2 = self.decode_head.module(all_out1)
        #1.1 loss
        loss_SAM_seg = self.decode_head.module.loss_by_feat(all_out2)
        #1.2
        loss_SAM_seg.backward()
        optim_wrapper.step()
       


    

    
    #add hdd 2025/2/28
    def extract_feat_SAM(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        image_embeddings = self.backbone.module.image_encoder(inputs)
       # return image_embeddings
        #add hdd 2025/3/3
        return [image_embeddings] if isinstance(image_embeddings, Tensor) else image_embeddings


    def SAM2_backbone_forward(self, img, B_img):
        outputs = dict()
        fb_s = self.extract_feat_SAM(img)
        fb_t = self.extract_feat_SAM(B_img)
        outputs['fb_s'] = fb_s
        outputs['fb_t'] = fb_t
        return outputs   
    














