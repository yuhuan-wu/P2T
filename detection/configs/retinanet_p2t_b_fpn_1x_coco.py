_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', '_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='p2t_base',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/p2t_base.pth'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5)
)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

total_epochs = 12
fp16 = None
find_unused_parameters = True