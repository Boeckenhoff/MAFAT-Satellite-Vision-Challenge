import os

from mmdet.apis import set_random_seed

CLASSES_ALL=  ('small_vehicle','bus','medium_vehicle','large_vehicle',
     'double_trailer_truck','small_aircraft','large_aircraft','small_vessel','medium_vessel','large_vessel',
     'heavy_equipment', 'container','pylon', )

def modify_mmdet_config(cfg, learning_rate=1e-3, max_epochs=10):
    # # Modify dataset type and path
    cfg.dataset_type = 'MAFATDataset'
    cfg.data_root = './data/patchified_dataset'

    cfg.img_norm_cfg = dict(type='Normalize',
        mean=[795.0878, 795.0878, 795.0878], std=[460.0955, 460.0955, 460.0955], to_rgb=True)

    cfg.data.train.type = 'MAFATDataset'
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.ann_file = 'annfiles'
    cfg.data.train.img_prefix = 'images'
    cfg.data.train.subset = 'data/split/train.csv'
    color_type='unchanged'

    cfg.data.test.type = 'MAFATDataset'
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = 'annfiles'
    cfg.data.test.img_prefix = 'images'
    cfg.data.test.subset = 'data/split/test.csv'

    cfg.data.val.type = 'MAFATDataset'
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.ann_file = 'annfiles'
    cfg.data.val.img_prefix = 'images'
    cfg.data.val.subset = 'data/split/test.csv'

    # edit pipeline to handle 16bit images
    cfg.train_pipeline[0].color_type = 'unchanged'
    cfg.train_pipeline[0].type = 'LoadImageFromFilePIL'
    cfg.test_pipeline[0].color_type = 'unchanged'
    cfg.test_pipeline[0].type = 'LoadImageFromFilePIL'

    cfg.data.train.pipeline[5] = cfg.img_norm_cfg
    cfg.data.val.pipeline[1]['transforms'][1] = cfg.img_norm_cfg
    cfg.data.test.pipeline[1]['transforms'][1] = cfg.img_norm_cfg

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head[0].num_classes = len(CLASSES_ALL)
    cfg.model.roi_head.bbox_head[1].num_classes = len(CLASSES_ALL)
    # Load pre-trainied weights, trained on DOTA
    cfg.load_from = os.path.join('./src/contents','redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth')

    # Set up working dir to save files and logs.
    cfg.work_dir = './models'

    cfg.optimizer.lr = learning_rate
    cfg.lr_config.warmup = None
    cfg.runner.max_epochs = max_epochs
    cfg.log_config.interval = 2

    cfg.data.samples_per_gpu=1
    cfg.data.workers_per_gpu=1

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 1

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device='cuda'

    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook'),
        dict(type='DVCLoggerHook')]
    
    # dump_file = "my_config_redet.py"
    # cfg.dump(dump_file)