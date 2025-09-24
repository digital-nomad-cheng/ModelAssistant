_base_ = ['swift_yolo_tiny_1xb16_300e_coco.py']

# KD specific additions
teacher = dict(
    cfg='work_dirs/swift_yolo_medium_1xb16_300e_coco/swift_yolo_medium_1xb16_300e_coco.py',
    checkpoint='work_dirs/swift_yolo_medium_1xb16_300e_coco/best_coco_bbox_mAP_epoch_5.pth',
    frozen=True,
    eval_bn=True,
)

kd = dict(
    start_epoch=0,
    temperature=4.0,
    obj_thr=0.30,
    max_pos_per_level=0,  # 0 means no cap
    feat_layers='auto',   # use backbone outputs
    losses=dict(
        cls=dict(type='KDLoss', temperature=4.0, loss_weight=1.0),
        obj=dict(type='KDLoss', use_sigmoid=True, loss_weight=1.0),
        bbox=dict(type='BBoxMSELoss', loss_weight=2.0),
        feat=dict(type='FeatureMSELoss', project_if_mismatch=True, loss_weight=1.0),
    ),
)
