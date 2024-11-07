# dataset settings
dataset_type = 'BDDDatasetMM'
data_root = ''

crop_size = (512, 512)
num_classes = 29
train_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs', meta_keys=['file_name',
                                          'ori_shape',
                                          'img_shape',
                                          'scale_factor',
                                          'reduce_zero_label'])
]
test_pipeline = [
    dict(type='AVResize', scale=crop_size, keep_ratio=False, test=True),
    dict(type='PackSegInputs', meta_keys=['file_name',
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
            DATASET_DIR = '/store/datasets/bdd100k/', 
            SYNTH_TRAIN_DIR = '/store/harsh/data/bdd_synthetic_ft_ceq', # Always set in the sub programs
            TRAIN_META_PATH = 'bdd100k/',
            VAL_META_PATH = 'bdd100k/',
            PALLETE = 'COCO'
        ),
        pipeline=train_pipeline,
        validation=False,
        segmentation=True,
        image_meta_data=True,
        serialize_data=True,
        mixing_ratio=0.5
        ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_config=dict(
            DATASET_DIR = '/store/datasets/bdd100k/', 
            SYNTH_TRAIN_DIR = None, # Always set in the sub programs
            TRAIN_META_PATH = 'bdd100k/',
            VAL_META_PATH = 'bdd100k/',
            PALLETE = 'COCO'
        ),
        pipeline=test_pipeline,
        validation=True,
        segmentation=True,
        image_meta_data=True,
        serialize_data=True,
        mixing_ratio=0.0
        ))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
