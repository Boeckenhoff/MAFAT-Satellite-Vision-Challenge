import argparse
import os.path as osp

from mmcv import Config

# Check MMRotate installation
import mmrotate
print(mmrotate.__version__)

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

import mmcv

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

def add_parser(parser):
    """Add arguments."""

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='path to the metadatafile')

    # argument for splitting tests
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=3,
        help='')
    
    parser.add_argument(
        '--seed',
        type=int,
        default= 42)

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting dataset')
    add_parser(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile('src/config.py')

    cfg.optimizer.lr = args.learning_rate
    cfg.runner.max_epochs = args.max_epochs

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    #datasets = [MAFAT_Dataset]

    # Build the detector
    model = build_detector( cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg= cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True) # doesn't work in colab

if __name__ == '__main__':
    main()