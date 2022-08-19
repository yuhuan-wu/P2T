_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/p2t_small.pth',
    backbone=dict(
        type='p2t_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150),
    )
cudnn_benchmark = False
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
data = dict(samples_per_gpu=2)
find_unused_parameters = True
