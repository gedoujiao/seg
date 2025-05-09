import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS

@MODELS.register_module()
class FusionNeck(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.fusion_conv = ConvModule(
            in_channels * 4, out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')
        )

    def forward(self, SAM_backbone_oup):
        # 通道拼接
        fused_feature = torch.cat([SAM_backbone_oup[0], SAM_backbone_oup[1], SAM_backbone_oup[2], SAM_backbone_oup[3]], dim=1)
        # 1×1 卷积融合
        fused_feature = self.fusion_conv(fused_feature)
        return fused_feature
