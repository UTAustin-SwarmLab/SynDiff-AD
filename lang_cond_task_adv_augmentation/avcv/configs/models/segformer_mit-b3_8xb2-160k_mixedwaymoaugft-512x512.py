_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../dataset/waymo_ds_aug_ft.py',
    '../default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
                    embed_dims=64,
                    num_heads=[1, 2, 5, 8],
                    num_layers=[3, 4, 18, 3]),
    decode_head=dict(num_classes=150,
                     in_channels=[64, 128, 320, 512]))



train_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=['context_name',
                                          'context_frame',
                                          'camera_id',
                                          'ori_shape',
                                          'img_shape',
                                          'scale_factor',
                                          'condition',
                                          'reduce_zero_label'])
]
test_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False, test=True),
    dict(type='PackSegInputs', meta_keys=['context_name',
                                          'context_frame',
                                          'camera_id',
                                          'ori_shape',
                                          'img_shape',
                                          'scale_factor',
                                          'condition',
                                          'reduce_zero_label'])
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]


train_dataloader = dict(
    batch_size=3,
    num_workers=3,
    dataset=dict(
        pipeline=train_pipeline,
        )
  )

val_dataloader = dict(
    batch_size=6,
    num_workers=6,
        dataset=dict(
        pipeline=test_pipeline,
    )
   )
test_dataloader = val_dataloader