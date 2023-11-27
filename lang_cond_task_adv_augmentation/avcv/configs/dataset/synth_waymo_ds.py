# dataset settings
dataset_type = 'SynthWaymoDatasetMM'
data_root = ''

crop_size = (960, 640)
num_classes = 29
train_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs', meta_keys=['context_name',
                                          'context_frame',
                                          'camera_id',
                                          'ori_shape',
                                          'img_shape',
                                          'scale_factor',
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
                                          'reduce_zero_label'])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_config=dict(
            DATASET_DIR = '/store/harsh/data/waymo_synthetic/'
        ),
        pipeline=train_pipeline,
        validation=False,
        segmentation=True,
        image_meta_data=True,
        serialize_data=True
        ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_config=dict(
            DATASET_DIR = '/store/harsh/data/waymo_synthetic/'
        ),
        pipeline=test_pipeline,
        validation=True,
        segmentation=True,
        image_meta_data=True,
        serialize_data=True
        ))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator