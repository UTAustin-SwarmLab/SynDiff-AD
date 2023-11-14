_base_ = ['mask2former_r50_8xb2-90k_waymo-960x640.py']
crop_size = (512, 512)

train_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False, test=True),
    dict(type='PackSegInputs')
]
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32)
    )

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
  )

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
   )