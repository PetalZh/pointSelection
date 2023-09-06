voxel_size = [0.25, 0.25, 8]
model = dict(
    type='MVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=[0.25, 0.25, 8],
            max_voxels=(30000, 40000))),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.25, 0.25, 8],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg=dict(type='naiveSyncBN1d', eps=0.001, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 400]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=0.001, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=0.001, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
                    [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                    [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                    [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                    [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                    [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                    [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965]],
            sizes=[[4.60718145, 1.95017717, 1.72270761],
                   [6.73778078, 2.4560939, 2.73004906],
                   [12.01320693, 2.87427237, 3.81509561],
                   [1.68452161, 0.60058911, 1.27192197],
                   [0.7256437, 0.66344886, 1.75748069],
                   [0.40359262, 0.39694519, 1.06232151],
                   [0.48578221, 2.49008838, 0.98297065]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
metainfo = dict(classes=[
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
])
dataset_type = 'NuScenesDataset'
data_root = '/data/nuscenes/full/'
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
backend_args = None
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        sample_method='grid_fps',
        octree_height=10,
        sample_rate=0.4,
        oc_node_capacity=100,
        backend_args=None),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, sample_method='grid_fps',
                octree_height=10,
                sample_rate=0.4,
                oc_node_capacity=100,backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        sample_method='grid_fps',
        octree_height=10,
        sample_rate=0.4,
        oc_node_capacity=100,
        backend_args=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        test_mode=True,
        sample_method='grid_fps',
        octree_height=10,
        sample_rate=0.4,
        oc_node_capacity=100,
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
                point_cloud_range=[-50, -50, -5, 50, 50, 3])
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        sample_method='grid_fps',
        octree_height=10,
        sample_rate=0.4,
        oc_node_capacity=100,
        backend_args=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        test_mode=True,
        sample_method='grid_fps',
        octree_height=10,
        sample_rate=0.4,
        oc_node_capacity=100,
        backend_args=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesDataset',
        data_root='/data/nuscenes/full/',
        ann_file='nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                sample_method='grid_fps',
                octree_height=10,
                sample_rate=0.4,
                oc_node_capacity=100,
                backend_args=None),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                sample_method='grid_fps',
                octree_height=10,
                sample_rate=0.4,
                oc_node_capacity=100,
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='Pack3DDetInputs',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        metainfo=dict(classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=False,
        data_prefix=dict(
            pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR',
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root='/data/nuscenes/full/',
        ann_file='nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                sample_method='grid_fps',
                octree_height=10,
                sample_rate=0.4,
                oc_node_capacity=100,
                backend_args=None),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                test_mode=True,
                sample_method='grid_fps',
                octree_height=10,
                sample_rate=0.4,
                oc_node_capacity=100,
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
                        point_cloud_range=[-50, -50, -5, 50, 50, 3])
                ]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        metainfo=dict(classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
        modality=dict(use_lidar=True, use_camera=False),
        data_prefix=dict(
            pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root='/data/nuscenes/full/',
        ann_file='nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                sample_method='grid_fps',
                octree_height=10,
                sample_rate=0.4,
                oc_node_capacity=100,
                backend_args=None),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                test_mode=True,
                sample_method='grid_fps',
                octree_height=10,
                sample_rate=0.4,
                oc_node_capacity=100,
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
                        point_cloud_range=[-50, -50, -5, 50, 50, 3])
                ]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        metainfo=dict(classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        data_prefix=dict(
            pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR',
        backend_args=None))
val_evaluator = dict(
    type='NuScenesMetric',
    data_root='/data/nuscenes/full/',
    ann_file='/data/nuscenes/full/nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='NuScenesMetric',
    data_root='/data/nuscenes/full/',
    ann_file='/data/nuscenes/full/nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=None)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=24)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]
auto_scale_lr = dict(enable=False, base_batch_size=32)
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
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
launcher = 'none'
work_dir = './work_dirs/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d'
