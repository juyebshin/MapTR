_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]




img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
fixed_ptsnum_per_gt = 400
sample_dist = 1.5
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.

# InstaGraM
use_dist_embed = True
find_unused_parameters = True

model = dict(
    type='InstaGraM',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='InstaGraMHead',
        num_classes=num_map_classes,
        bev_h=bev_h_,
        bev_w=bev_w_,
        voxel_size=voxel_size,
        as_two_stage=False,
        sample_dist=sample_dist,
        transform_method='minmax',
        transformer=dict(
            type='InstaGraMPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
                ), # encoder
            decoder=dict(
                type='InstaGraMDecoder',
                vertex_decoder=dict(
                    type='ElementDecoder',
                    bev_encoder=dict(
                        type='ResNet',
                        depth=18,
                        in_channels=_dim_,
                        base_channels=_pos_dim_,
                        num_stages=1,
                        strides=(1,),
                        dilations=(1,),
                        out_indices=(0,),
                        ),
                    conv_cfg=dict(
                        type='Conv2d',
                        in_channels=_pos_dim_,
                        out_channels=65, # class-wise: num_map_classes, else: 65
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                    ),
                distance_decoder=dict(
                    type='ElementDecoder',
                    bev_encoder=dict(
                        type='ResNet',
                        depth=18,
                        in_channels=_dim_,
                        base_channels=_pos_dim_,
                        num_stages=1,
                        strides=(1,),
                        dilations=(1,),
                        out_indices=(0,),
                        ),
                    conv_cfg=dict(
                        type='Conv2d',
                        in_channels=_pos_dim_,
                        out_channels=num_map_classes if use_dist_embed else _dim_,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                    bev_decoder=dict(
                        upsample_cfg=dict(
                            type='bilinear',
                            scale_factor=2,
                            align_corners=True,
                            ),
                        plugin_cfg=dict(
                            type='ConvModule',
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        in_channels=_pos_dim_,
                        blocks=(128, 128, _pos_dim_),
                    ) if use_dist_embed else None,
                    ),
                graph_encoder=dict(
                    type='GraphEncoder',
                    embed_dim=_dim_,
                    layers=[_pos_dim_//2, _pos_dim_, _dim_],
                    norm_type='BN1d',
                    ),
                attn_cfg=dict(
                    type='AttentionalGNN',
                    embed_dim=_dim_,
                    norm_type='BN1d',
                ),
                num_classes=num_map_classes,
                cell_size=8,
                use_dist_embed=use_dist_embed,
                # pc_range=point_cloud_range,
                # voxel_size=voxel_size,
                dist_thr=10.0,
                vertex_thr=0.01,
                num_vertices=fixed_ptsnum_per_gt,
                embed_dim=_dim_,
                pos_freq=10,
                sinkhorn_iters=100,
                num_gnn_layers=9,
                ), # decoder
            ), # transformer
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_vtx=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,), # 2.0
        loss_dtm=dict(
            type='MSELoss',
            loss_weight=1.0,), # 5.0
        loss_graph=dict(
            type='GraphLoss',
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            cdist_thr=1.5,
            reduction='mean',
            loss_weight=dict(cls=0.01, match=0.001)), # 0.1, 0.005
    ), # pts_bbox_head
    # model training and testing settings
    # train_cfg=dict(pts=dict(
    #     grid_size=[512, 512, 1],
    #     voxel_size=voxel_size,
    #     point_cloud_range=point_cloud_range,
    #     out_size_factor=4,
    #     assigner=dict(
    #         type='MapTRAssigner',
    #         cls_cost=dict(type='FocalLossCost', weight=2.0),
    #         reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
    #         # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
    #         # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
    #         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
    #         pts_cost=dict(type='OrderedPtsL1Cost', 
    #                   weight=5),
    #         pc_range=point_cloud_range)))
    )

dataset_type = 'CustomNuScenesLocalMapDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        sample_dist=sample_dist,
        num_samples=fixed_ptsnum_per_gt,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             map_ann_file=data_root + 'nuscenes_map_vector_anns_val.json',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             pc_range=point_cloud_range,
             fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
             sample_dist=sample_dist,
             num_samples=fixed_ptsnum_per_gt,
             eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
             padding_value=-10000,
             map_classes=map_classes,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              map_ann_file=data_root + 'nuscenes_map_vector_anns_val.json',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              pc_range=point_cloud_range,
              fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
              sample_dist=sample_dist,
              num_samples=fixed_ptsnum_per_gt,
              eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
              padding_value=-10000,
              padding=True,
              map_classes=map_classes,
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='Adam',
    lr=1e-3,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-7)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
# lr_config = dict(
#     policy='step',
#     # warmup='linear',
#     # warmup_iters=500,
#     # warmup_ratio=1.0 / 3,
#     step=10,
#     gamma=0.1)
total_epochs = 24
# total_epochs = 50
# evaluation = dict(interval=1, pipeline=test_pipeline)
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=2) # interval=5