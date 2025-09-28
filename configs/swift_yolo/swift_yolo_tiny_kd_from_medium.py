_base_ = ['swift_yolo_tiny_1xb16_300e_coco.py']

# KD specific additions
teacher = dict(
    cfg='work_dirs/swift_yolo_medium_1xb16_300e_coco/swift_yolo_medium_1xb16_300e_coco.py',
    checkpoint='work_dirs/swift_yolo_medium_1xb16_300e_coco/best_coco_bbox_mAP_epoch_290.pth',
    frozen=True,
    eval_bn=True,
)

kd = dict(
    start_epoch=0,
    # For CWD we only need features; head logits KD are removed based on paper focus.
    feat_layers='auto',   # using backbone outputs by default
    losses=dict(
        # method='spatial' implements paper's per-channel spatial distribution KL.
        # High loss_weight=10.0 retained from your edit; tune if instability occurs.
        cwd=dict(type='ChannelWiseDistillLoss', tau=2.0, method='spatial', loss_weight=10.0),
    ),
)
