# dataset settings
dataset_type = 'WaymoDatasetMM'
data_root = ''

crop_size = (512, 512 )
num_classes = 29
train_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_config=dict(
            TRAIN_DIR = '/store/datasets/waymo/training/', 
            EVAL_DIR = '/store/datasets/waymo/validation/',
            TEST_SET_SOURCE = '/content/waymo-open-dataset/tutorial/2d_pvps_validation_frames.txt',
            SAVE_FRAMES = [0,1,2]),
        pipeline=train_pipeline,
        validation=False,
        segmentation=True,
        image_meta_data=True,
        serialize_data=False
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_config=dict(
            TRAIN_DIR = '/store/datasets/waymo/training/', 
            EVAL_DIR = '/store/datasets/waymo/validation/',
            TEST_SET_SOURCE = '/content/waymo-open-dataset/tutorial/2d_pvps_validation_frames.txt',
            SAVE_FRAMES = [0,1,2]),
        pipeline=train_pipeline,
        validation=True,
        segmentation=True,
        image_meta_data=True,
        serialize_data=False
        ))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator