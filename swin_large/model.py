# =============================================================================
# SWIN TRANSFORMER LARGE - MODEL DEFINITION
# =============================================================================
# MMDetection-based Swin Transformer Large with Cascade RCNN for detection
# With optional class-balanced loss weights for imbalanced data
# =============================================================================

import os

from config import USE_CLASS_WEIGHTS, CLASS_WEIGHTS


def get_class_balanced_loss_config():
    """Generate loss config with class weights if enabled."""
    if USE_CLASS_WEIGHTS:
        # Convert class weights dict to list format
        weights = [CLASS_WEIGHTS.get(i, 1.0) for i in range(8)]
        return {
            'use_class_weights': True,
            'class_weight': weights
        }
    return {'use_class_weights': False}


def get_mmdet_config_str(
    work_dir,
    train_json,
    val_json,
    train_img_prefix,
    val_img_prefix,
    class_names,
    use_class_weights=False,
    class_weights=None
):
    """
    Generate MMDetection config string for Swin Transformer Large.
    
    Swin-Large specifications:
    - embed_dims: 192 (vs 96 for Tiny)
    - depths: [2, 2, 18, 2] (deeper stage 3)
    - num_heads: [6, 12, 24, 48]
    - Pretrained on ImageNet-22K
    """
    
    classes_str = str(tuple(class_names))
    
    # Class-balanced Focal Loss config
    if use_class_weights and class_weights:
        # Calculate alpha per class (inverse of frequency)
        # Normalize weights to sum to 1
        total_weight = sum(class_weights.values())
        weight_values = [1.0] + [class_weights.get(i, 1.0) / total_weight for i in range(8)]  # +1 for background
        weight_str = str(weight_values)
        
        # Use CrossEntropyLoss with per-class weights (FocalLoss only supports sigmoid mode)
        loss_cls_config = f'''dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    class_weight={weight_str},
                    loss_weight=1.0
                )'''
    else:
        # Standard Focal Loss
        loss_cls_config = '''dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0
                )'''
    
    config_str = f'''
# =============================================================================
# Swin Transformer Large Configuration for Cervical Cancer Detection
# =============================================================================
# Class-balanced loss: {use_class_weights}
# =============================================================================

_base_ = [
    '/root/.cache/mim/mmdet/configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '/root/.cache/mim/mmdet/configs/_base_/default_runtime.py'
]

# =============================================================================
# SWIN TRANSFORMER LARGE BACKBONE
# =============================================================================

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0)
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=True,
                loss_cls={loss_cls_config},
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]
                ),
                reg_class_agnostic=True,
                loss_cls={loss_cls_config},
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]
                ),
                reg_class_agnostic=True,
                loss_cls={loss_cls_config},
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            )
        ]
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001),
            max_per_img=300
        )
    )
)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

dataset_type = 'CocoDataset'
data_root = '{work_dir}/'

CLASS_NAMES = {{
    0: 'NILM', 1: 'ENDO', 2: 'INFL', 3: 'ASCUS',
    4: 'LSIL', 5: 'HSIL', 6: 'ASCH', 7: 'SCC'
}}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1024, 1024), (1152, 1152), (1280, 1280), (1408, 1408)],
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.85, 0.85),
        allow_negative_crop=False
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(
        type='RandomAffine',
        max_rotate_degree=15.0,
        max_shear_degree=5.0,
        scaling_ratio_range=(0.9, 1.1),
        border=(0, 0)
    ),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{train_json}',
        data_prefix=dict(img='{train_img_prefix}'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes={classes_str})
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{val_json}',
        data_prefix=dict(img='{val_img_prefix}'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes={classes_str})
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '{val_json}',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# =============================================================================
# OPTIMIZER
# =============================================================================

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=48,
        by_epoch=True,
        eta_min=1e-6,
        T_max=48
    )
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, priority='ABOVE_NORMAL')
]

default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
'''
    
    return config_str


def save_mmdet_config(output_path, **kwargs):
    """Save MMDetection config to file"""
    # Add class weights from config if enabled
    kwargs['use_class_weights'] = USE_CLASS_WEIGHTS
    kwargs['class_weights'] = CLASS_WEIGHTS if USE_CLASS_WEIGHTS else None
    
    config_str = get_mmdet_config_str(**kwargs)
    
    with open(output_path, 'w') as f:
        f.write(config_str)
    
    print(f"Saved MMDetection config to: {output_path}")
    if USE_CLASS_WEIGHTS:
        print("  Class-balanced loss: ENABLED")
    return output_path


if __name__ == "__main__":
    from config import (
        OUTPUT_DIR, WORK_DIR, CLASS_NAMES,
        TRAIN_IMG_DIR, VAL_IMG_DIR
    )
    
    config_path = save_mmdet_config(
        output_path=os.path.join(OUTPUT_DIR, 'swin_large_config.py'),
        work_dir=OUTPUT_DIR,
        train_json='annotations_train.json',
        val_json='annotations_val.json',
        train_img_prefix=TRAIN_IMG_DIR,
        val_img_prefix=VAL_IMG_DIR,
        class_names=CLASS_NAMES
    )
    print(f"Config saved to: {config_path}")
