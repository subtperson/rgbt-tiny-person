dataset_type = 'DronePerson'
data_root = '/media/vision/lzy/data/RGBTDronePerson/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
    std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]))
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='MultiNormalize',
        mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
        std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='MultiNormalize',
                mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
                std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='DronePerson',
        ann_file='/media/vision/lzy/data/RGBTDronePerson/train_thermal.json',
        img_prefix='/media/vision/lzy/data/RGBTDronePerson/train/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='MultiNormalize',
                mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
                std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='DronePerson',
        ann_file='/media/vision/lzy/data/RGBTDronePerson/val_thermal.json',
        img_prefix='/media/vision/lzy/data/RGBTDronePerson/val/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([115.37, 121.82,
                                    122.63], [93.1, 93.1, 93.1]),
                        std_list=([85.13, 89.01, 88.27], [50.24, 50.24,
                                                          50.24]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='DronePerson',
        ann_file='/media/vision/lzy/data/RGBTDronePerson/val_thermal.json',
        img_prefix='/media/vision/lzy/data/RGBTDronePerson/val/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([115.37, 121.82,
                                    122.63], [93.1, 93.1, 93.1]),
                        std_list=([85.13, 89.01, 88.27], [50.24, 50.24,
                                                          50.24]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='QFDet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSQHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    bbox_prehead=dict(
        type='QFDetPreHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
        loss_quality=dict(type='MSELoss', loss_weight=0.5)),
    base_fusion='cat',
    quality_attention=True,
    poolupsample=1,
    train_cfg=dict(
        assigner=dict(
            type='QLSAssigner',
            topk=9,
            alpha=0.8,
            quality='x',
            iou_calculator=dict(type='BboxDistanceMetric'),
            iou_mode='siwd',
            overlap_mode='hybrid'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))
work_dir = 'work_dir/qfdet_star_r50_fpn/rgbtdroneperson/20240222_221733'
auto_resume = False
gpu_ids = [0]
