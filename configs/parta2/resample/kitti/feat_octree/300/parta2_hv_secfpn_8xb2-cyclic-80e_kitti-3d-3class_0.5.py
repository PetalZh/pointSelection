lr = 0.0018
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=16,
        eta_min=0.018,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=1.8e-07,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=16,
        eta_min=0.8947368421052632,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=24,
        eta_min=1,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
model = dict(
    type='PartA2',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            voxel_size=[0.05, 0.05, 0.1],
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseUNet',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    rpn_head=dict(
        type='PartA2RPNHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    roi_head=dict(
        type='PartAggregationROIHead',
        num_classes=3,
        semantic_head=dict(
            type='PointwiseSemanticHead',
            in_channels=16,
            extra_width=0.2,
            seg_score_thr=0.3,
            num_classes=3,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_part=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)),
        seg_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='max')),
        bbox_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='avg')),
        bbox_head=dict(
            type='PartA2BboxHead',
            num_classes=3,
            seg_in_channels=16,
            part_in_channels=4,
            seg_conv_channels=[64, 64],
            part_conv_channels=[64, 64],
            merge_conv_channels=[128, 128],
            down_conv_channels=[128, 256],
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            shared_fc_channels=[256, 512, 512, 512],
            cls_channels=[256, 256],
            reg_channels=[256, 256],
            dropout_ratio=0.1,
            roi_feat_size=14,
            with_corner_loss=True,
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=0.1111111111111111,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=9000,
            nms_post=512,
            max_num=512,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=False),
        rcnn=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1024,
            nms_post=100,
            max_num=100,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.01,
            score_thr=0.1)))
dataset_type = 'KittiDataset'
data_root = '/data/kitti_detection_3d/'  # /data/kitti_detection_3d/
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
db_sampler = dict(
    data_root='/data/kitti_detection_3d/',
    info_path='/data/kitti_detection_3d/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=['Pedestrian', 'Cyclist', 'Car'],
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sample_method='feat_octree',
        sample_rate=0.5,
        oc_node_capacity=300,
        backend_args=None),
    backend_args=None)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sample_method='feat_octree',
        sample_rate=0.5,
        oc_node_capacity=300,
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='/data/kitti_detection_3d/',
            info_path='/data/kitti_detection_3d/kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
            classes=['Pedestrian', 'Cyclist', 'Car'],
            sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sample_method='feat_octree',
                sample_rate=0.5,
                oc_node_capacity=300,
                backend_args=None),
            backend_args=None)),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(type='ObjectNameFilter', classes=['Pedestrian', 'Cyclist', 'Car']),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sample_method='feat_octree',
        sample_rate=0.5,
        oc_node_capacity=300,
        backend_args=None),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -40, -3, 70.4, 40, 1])
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        sample_method='feat_octree',
        sample_rate=0.5,
        oc_node_capacity=300,
        backend_args=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='/data/kitti_detection_3d/',
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    sample_method='feat_octree',
                    sample_rate=0.5,
                    oc_node_capacity=300,
                    backend_args=None),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='ObjectSample',
                    db_sampler=dict(
                        data_root='/data/kitti_detection_3d/',
                        info_path='/data/kitti_detection_3d/kitti_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(
                                Car=5, Pedestrian=10, Cyclist=10)),
                        classes=['Pedestrian', 'Cyclist', 'Car'],
                        sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
                        points_loader=dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4,
                            sample_method='feat_octree',
                            sample_rate=0.5,
                            oc_node_capacity=300,
                            backend_args=None),
                        backend_args=None)),
                dict(
                    type='ObjectNoise',
                    num_try=100,
                    translation_std=[1.0, 1.0, 0.5],
                    global_rot_range=[0.0, 0.0],
                    rot_range=[-0.78539816, 0.78539816]),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(
                    type='ObjectNameFilter',
                    classes=['Pedestrian', 'Cyclist', 'Car']),
                dict(type='PointShuffle'),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            metainfo=dict(classes=['Pedestrian', 'Cyclist', 'Car']),
            box_type_3d='LiDAR',
            test_mode=False,
            backend_args=None)))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='/data/kitti_detection_3d/',
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sample_method='feat_octree',
                sample_rate=0.5,
                oc_node_capacity=300,
                backend_args=None),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -40, -3, 70.4, 40, 1])
                ]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        metainfo=dict(classes=['Pedestrian', 'Cyclist', 'Car']),
        box_type_3d='LiDAR',
        test_mode=True,
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='/data/kitti_detection_3d/',
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sample_method='feat_octree',
                sample_rate=0.5,
                oc_node_capacity=300,
                backend_args=None),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        metainfo=dict(classes=['Pedestrian', 'Cyclist', 'Car']),
        box_type_3d='LiDAR',
        test_mode=True,
        backend_args=None))
val_evaluator = dict(
    type='KittiMetric',
    ann_file='/data/kitti_detection_3d/kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='KittiMetric',
    ann_file='/data/kitti_detection_3d/kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None)
find_unused_parameters = True
launcher = 'none'
work_dir = './work_dirs/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class'
