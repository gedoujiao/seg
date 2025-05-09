_base_ = [
    '../configs/_base_/models/upernet_vit-b16_ln_mln.py',
    '../configs/_base_/datasets/uda_isprs.py',
    '../configs/_base_/default_runtime.py', './schedule_20k_multioptimizers.py'
]
crop_size = (512, 512)
data_preprocessor = dict(type='SegDataPreProcessor_UDA', size=crop_size)

## added by LYU: 2024/07/10
arch = 'base'
'''
SAM_dict = dict(
    type='SAM',
    image_encoder_cfg=dict(
        type='mmpretrain.ViTSAM',
        arch=arch,
        img_size=crop_size[0],
        patch_size=16,
        out_channels=256,
        out_indices=(2, 5, 8, 11),
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
    ),
    
    prompt_encoder_cfg=dict(
        type='PromptEncoder',
        embed_dim=256,
        image_embedding_size=(crop_size[0]//16, crop_size[0]//16),
        input_image_size=crop_size,
        mask_in_chans=16,
    ),
    mask_decoder_cfg=dict(
        type='MaskDecoder',
        num_multimask_outputs=3,
        transformer=dict(
            type='TwoWayTransformer',
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
            ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ))
'''
SAM_dict = dict(
    type='SAM2Base',
    image_encoder=dict(
        type='ImageEncoder',
        scalp=1,
        trunk=dict(
            type='Hiera',
            embed_dim=112,
            num_heads=2
        ),
        neck=dict(
            type='FpnNeck',
            position_encoding=dict(
                type='PositionEmbeddingSine',
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10000
            ),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[2, 3],  # output level 0 and 1 directly use the backbone features
            fpn_interp_model='nearest'
        )
    ),
    memory_attention=dict(
        type='MemoryAttention',
        d_model=256,
        pos_enc_at_input=True,
        layer=dict(
            type='MemoryAttentionLayer',
            activation='relu',
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            self_attention=dict(
                type='RoPEAttention',
                rope_theta=10000.0,
                feat_sizes=[64, 64],
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1
            ),
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            cross_attention=dict(
                type='RoPEAttention',
                rope_theta=10000.0,
                feat_sizes=[64, 64],
                rope_k_repeat=True,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64
            )
        ),
        num_layers=4
    ),
    
    memory_encoder=dict(
        type='MemoryEncoder',
        out_dim=64,
        position_encoding=dict(
            type='PositionEmbeddingSine',
            num_pos_feats=64,
            normalize=True,
            scale=None,
            temperature=10000
        ),
        mask_downsampler=dict(
            type='MaskDownSampler',
            kernel_size=3,
            stride=2,
            padding=1
        ),
        fuser=dict(
            type='Fuser',
            layer=dict(
                type='CXBlock',
                dim=256,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True  # depth-wise convs
            ),
            num_layers=2
        )
    ),
    num_maskmem=0,
    ##image_size = 1024
    image_size=512,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    use_high_res_features_in_sam=False,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=False,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    compile_image_encoder=False
)


checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoderwithSAMPromptSTAdv',
    data_preprocessor=data_preprocessor,
    backbone=None,
    SAM_config=SAM_dict,
    SAM_arch=arch,
    
   # fusion_neck=dict(
   # type='FusionNeck',
   # in_channels=256,
   # out_channels=256),
    
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[256, 256, 256, 256],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(type='UPerAttHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.15, 1.05, 1.25, 1.5])),
    auxiliary_head=None,
    discriminator_P=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=512),
     discriminator_S=dict(
        type='AdapSegDiscriminator',
        num_conv=3,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=6),
    SAM_EMA = dict(
        training_ratio=0.75,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=[1.01, 1.01, 1.51, 1.25, 2.01, 2.50],
        neck_EMA=dict(
            type='MultiLevelNeck',
            in_channels=[256, 256, 256, 256],
            out_channels=768,
            scales=[4, 2, 1, 0.5]
            ),
        decode_head_EMA=dict(
            type='UPerAttHead',
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.15, 1.05, 1.25, 1.5]))
    ),
    Prompt_EMA=dict(
        training_ratio=0.75,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=[1.01, 1.01, 1.51, 1.25, 2.01, 2.50],
        backbone_EMA=dict(
            type='MixVisionTransformer',
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 6, 40, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),
        decode_head_EMA=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.15, 1.05, 1.25, 1.25]))
    ),
    Prompt_backbone=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    Prompt_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.15, 1.05, 1.25, 1.25])),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

#### Optimizer of SAM Segmentor
optim_wrapper_backbone = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optim_wrapper_neck = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))

optim_wrapper_head = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))

optim_wrapper_discriminator_P = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0001, betas=(0.9, 0.99), weight_decay=0.01))
optim_wrapper_discriminator_S = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0001, betas=(0.9, 0.99), weight_decay=0.01))
#### Optimizer of Prompt segmentor
optim_wrapper_Prompt_backbone = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optim_wrapper_Prompt_head = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))
##add hdd 3/14
#optim_wrapper_fusion_neck = dict(
 #   type='OptimWrapper',
 #   optimizer=dict(
 #       type='AdamW', lr=0.00006, betas=(0.9, 0.999),  weight_decay=0.01))


optim_wrapper = dict(
    backbone=optim_wrapper_backbone,
    neck=optim_wrapper_neck,
    decode_head=optim_wrapper_head,
    discriminator_P=optim_wrapper_discriminator_P,
    discriminator_S=optim_wrapper_discriminator_S,
    Prompt_backbone=optim_wrapper_Prompt_backbone,
    Prompt_head=optim_wrapper_Prompt_head,
    ##add hdd 3/14
 #   fusion_neck=optim_wrapper_fusion_neck  
)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000))

train_dataloader = dict(batch_size=3, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
work_dir = './experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_results/'

find_unused_parameters=True

model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=find_unused_parameters)
'''
## Vaihingen_IRRG -> Potsdam IRRG
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='Vaihingen_IRRG_DA/img_dir/train', seg_map_path='Vaihingen_IRRG_DA/ann_dir/train'),
        ann_file='Vaihingen_IRRG_DA/train.txt',
        B_img_path='Potsdam_IRRG_DA/img_dir/train',
        B_img_file='Potsdam_IRRG_DA/train.txt'))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='Potsdam_IRRG_DA/img_dir/val', seg_map_path='Potsdam_IRRG_DA/ann_dir/val'),
        ann_file='Potsdam_IRRG_DA/val.txt'))
test_dataloader = val_dataloader
'''
'''
## Vaihingen_IRRG -> Potsdam RGB
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='Vaihingen_IRRG_DA/img_dir/train', seg_map_path='Vaihingen_IRRG_DA/ann_dir/train'),
        ann_file='Vaihingen_IRRG_DA/train.txt',
        B_img_path='Potsdam_RGB_DA/img_dir/train',
        B_img_file='Potsdam_RGB_DA/train.txt'))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='Potsdam_RGB_DA/img_dir/val', seg_map_path='Potsdam_RGB_DA/ann_dir/val'),
        ann_file='Potsdam_RGB_DA/val.txt'))
test_dataloader = val_dataloader
'''
'''
## Potsdam RGB -> Vaihingen_IRRG 
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='Potsdam_RGB_DA/img_dir/train', seg_map_path='Potsdam_RGB_DA/ann_dir/train'),
        ann_file='Potsdam_RGB_DA/train.txt',
        B_img_path='Vaihingen_IRRG_DA/img_dir/train',
        B_img_file='Vaihingen_IRRG_DA/train.txt'))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='Vaihingen_IRRG_DA/img_dir/val', seg_map_path='Vaihingen_IRRG_DA/ann_dir/val'),
        ann_file='Vaihingen_IRRG_DA/val.txt'))
test_dataloader = val_dataloader
'''