import argparse
import os
from model import model
import numpy as np
from mmrotate.core import eval_rbbox_map,  poly2obb_np
from mmrotate.apis import inference_detector_by_patches
import pandas as pd
import cv2
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Path to test dataset')
    parser.add_argument(
        '--test-dataset',
        type=str,
        default="data/split/test.csv",
        help='')

    parser.add_argument(
        '--model-config',
        type=str,
        default="model/mod_cfg.py",
        help='')


    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default="model/epoch_3.pth",
        help='')

    args = parser.parse_args()
    return args

def get_og_testdataset(test_csv, classes):
    with open(test_csv) as f:
        filenames = f.read().splitlines() 
    
    for filename in filenames:
        img = cv2.imread(f"data/labeled_dataset/images/{filename}.tiff", -1)
        meta = pd.read_csv("data/metadata_train.csv")
        meta = meta[ meta.Frame == filename]
        meta = meta[["Resolution", "Sun_Elevation", "Azimuth", "Sun_Azimuth"]]
        with open(f"data/labeled_dataset/labelTxt/{filename}.txt") as f:
            anns = f.read().splitlines() 
        bboxes = [poly2obb_np(np.array(ann.split()[:8], dtype=np.float32), version="le90") for ann in anns]
        bboxes = np.array(bboxes) if bboxes else np.empty(shape=(0,5))
        labels  = np.array([classes.index(ann.split()[8]) for ann in anns])
        anns = {"bboxes": bboxes, "labels": labels}
        yield img, meta, anns


def convert_pre_polys_to_bboxs(polys):
    if not polys:
        return np.empty(shape=(0,6))
    bboxes =  np.vstack([np.hstack((poly2obb_np( np.float32(poly[1:]), version="le90"), poly[0])) for poly in polys])
    return bboxes

def main():
    args = parse_args()

    model_ins = model()
    model_ins.checkpoint_path = args.model_checkpoint
    model_ins.config_path =  args.model_config
    model_ins.load(os.getcwd())

    for img, meta, _ in  get_og_testdataset(args.test_dataset, model_ins.CLASSES):
        model_ins.collect_statistics_iter(img, meta)

    model_ins.update_model()

    annotations = []
    predictions = []
    for img, meta, ann in  get_og_testdataset(args.test_dataset, model_ins.CLASSES):
        prediction = model_ins.predict(img, meta)
        predictions.append([convert_pre_polys_to_bboxs(cls_pred) for cls_pred in prediction])
        annotations.append(ann)

    mean_ap, _ = eval_rbbox_map(
        predictions,
        annotations,
    )
    eval_results = {'mAP': mean_ap}



if __name__ == '__main__':
    main()
