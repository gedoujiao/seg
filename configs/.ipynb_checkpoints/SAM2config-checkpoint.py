_base_ = [
    '../configs/_base_/models/upernet_vit-b16_ln_mln.py',
    '../configs/_base_/datasets/uda_isprs.py',
    '../configs/_base_/default_runtime.py', './schedule_20k_multioptimizers.py'
]
crop_size = (512, 512)
data_preprocessor = dict(type='SegDataPreProcessor_UDA', size=crop_size)

arch = 'large'
SAM2_dict = dict(
    type='SAM2Base',
    image_encoder=dict(
        type='ImageEncoder',
        scalp=1,
        trunk=dict(
            type='Hiera',
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8]
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
            backbone_channel_list=[1152, 576, 288, 144],
            fpn_top_down_levels=[2, 3],
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
                feat_sizes=[32, 32],
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
                feat_sizes=[32, 32],
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
                use_dwconv=True
            ),
            num_layers=2
        )
    ),
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    use_high_res_features_in_sam=True,
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
    type='SAM2PromptSTAdv',
    data_preprocessor=data_preprocessor,
    backbone=None,
    SAM_config=SAM_dict,
    SAM_arch=arch,
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




