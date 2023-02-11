import argparse
from model import model
from mmrotate.core import eval_rbbox_map
import pandas as pd
import cv2

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting dataset')
    parser.add_argument(
        '--test-dataset',
        type=str,
        default="data/split/test",
        help='')

    parser.add_argument(
        '--model-config',
        type=str,
        default="checkpoints/config.py",
        help='')


    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default="checkpoints/epoch_1.pth",
        help='')

    args = parser.parse_args()
    return args

def get_og_testdataset(test):
    with open(test) as f:
        filenames = f.read().splitlines() 
    
    for filename in filenames:
        img = cv2.imread("data/labeled_dataset/images/{filename}.tiff", -1)
        meta = pd.read_csv("data/metadata_train.csv")
        meta = meta[ meta.Frame == filename]
        meta = meta[["Resolution", "Sun_Elevation", "Azimuth", "Sun_Azimuth"]]
        with open("data/labeled_dataset/lblText/{filename}.txt") as f:
            ann = f.read().splitlines() 
        yield img, meta, ann


def main():
    args = parse_args()

    model = model()
    model.checkpoint_path = args.model_checkpoint
    model.config_path =  args.model_config
    model.load()

    for img, meta in  get_og_testdataset(args.test_dataset):
        model.collect_statistics_iter(img, meta)

    model.update_model()

    for img, meta in  get_og_testdataset(args.test_dataset):
        predictions = model.predict(img, meta)
    
    annotations = []
    mean_ap, _ = eval_rbbox_map(
        predictions,
        annotations,
    )
    eval_results = {'mAP': mean_ap}



if __name__ == '__main__':
    main()
