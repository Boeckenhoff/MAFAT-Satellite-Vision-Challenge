import mmcv
from mmcv.runner import load_checkpoint

from mmrotate.apis import inference_detector_by_patches
from mmrotate.models import build_detector

import os
import math
import numpy as np
from zipfile import ZipFile

class model():
    def __init__(self):
        self.make_initial_pass_over_dataset = True
        self.CLASSES = ('small_vehicle','bus','medium_vehicle','large_vehicle',
     'double_trailer_truck','small_aircraft','large_aircraft','small_vessel','medium_vessel','large_vessel',
     'heavy_equipment', 'container','pylon', ) # MAFAT classes
        self.stats = []
        self.checkpoint_path = r"../models/initial_submission.pth"
        self.config_path = r"config.py"
        # Initialize variables to store the running sum and square sum of pixel values
        self.running_sum = 0
        self.running_square_sum = 0
        # Initialize a variable to store the number of images processed
        self.num_images = 0
    
    
    def collect_statistics_iter(self, img, meta):
        '''img is a grayscale 16bit image of size 1280x1280 pixels (as numpy array).
        
        
        metadata is a dictionary with the metadata for the given image
        (similar to training phase, but without AOI and Hermetic data)
        '''
        # you can do something with the meta data
        
        # calculate mean and std using welford method        
        self.mean_and_std(img) 


    def update_model(self):
        # get mean and std
        self.std *= 2 ** 16
        self.mean *= 2 ** 16
        
        print("mean: %f, std: %f" % (self.mean, self.std))
        self.stats.append(self.mean)
        self.stats.append(self.std)
        
        # update test pipeline with computed statistics
        self.model.cfg.data.test.pipeline[1]['transforms'][1]['mean'] = [self.stats[0]]*3
        self.model.cfg.data.test.pipeline[1]['transforms'][1]['std'] = [self.stats[1]]*3       
    

    def mean_and_std(self, image):
        
        # Update the running sum and square sum with the new image data
        image = image / 2 ** 16
        self.running_sum += image.sum()
        self.running_square_sum += (image ** 2).sum()

        # Increment the number of images processed
        self.num_images += 1

        # Compute the mean and standard deviation on the fly
        self.mean = self.running_sum / (self.num_images * image.shape[0] * image.shape[1])
        self.std = np.sqrt((self.running_square_sum / (self.num_images * image.shape[0] * image.shape[1])) - self.mean**2)
         
    
    def load(self, dir_path):
        """loads the model.
        dir_path is for internal use only - do not remove it.
        all other paths, such as self.config_path should only contain the file name (in this case: my_config_redet.py).
        these paths must be concatenated with dir_path - see example in the code below
        make sure these files are in the same directory as the model.py file

        Args:
            dir_path (string): path to the submission directory (for internal use only).
        """        
        # check if directory contains a zip file, i.e. the model weights were downloaded from google drive
        files = os.listdir(dir_path)
        for filename in files:
            if filename.endswith("zip"):
                with ZipFile(os.path.join(dir_path, filename), 'r') as z:
                    z.extractall(dir_path)
                break
            
        # join paths
        config_path = os.path.join(dir_path, self.config_path)
        checkpoint_path = os.path.join(dir_path, self.checkpoint_path)
        # Set the device to be used for evaluation
        self.device='cuda:0'

        # Load the config
        self.config = mmcv.Config.fromfile(config_path)
        # adjust number of classes
        # self.config.model.bbox_head.num_classes = len(CLASSES_ALL)
        # Set pretrained to be None since we do not need pretrained model here
        self.config.model.pretrained = None

        # Initialize the detector
        self.model = build_detector(self.config.model)

        # Load checkpoint
        self.checkpoint = load_checkpoint(self.model, checkpoint_path, map_location=self.device)

        # Set the classes of models for inference
        self.model.CLASSES = self.checkpoint['meta']['CLASSES']

        # We need to set the model's cfg for inference
        self.model.cfg = self.config

        # Convert the model to GPU
        self.model.to(self.device)
        # Convert the model into evaluation mode
        self.model.eval()
        
    def predict(self, img, metadata=None):
        '''img is a grayscale 16bit image of size 1280x1280 pixels.
            expected output is a list of lists. each sub list represents all the detections for a class.
            the sub lists need to be in the same order as self.CLASSES.
            the detections need to be in the form of:
            confidence x1 y1 x2 y2 x3 y3 x4 y4
            
            metadata is a dictionary with the metadata for the given image
            (similar to training phase, but without AOI and Hermetic data)
        '''
            
        # the DNN expects a RGB image, we duplicate the grayscale image 3 times
        img = np.stack((img, )*3, axis=-1)
        # pass the image through the model to get detections
        result = inference_detector_by_patches(self.model, img, [640], [320], [1.0], 0.3)
        # result = inference_detector(self.model, img)
        for i, res in enumerate(result):
            result[i] = self.convert_rbb2polygon(res)
        return result
    
    def convert_rbb2polygon(self, bboxes):
        polygons = []
        for i, bbox in enumerate(bboxes):
            xc, yc, w, h, ag, conf = bbox
            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
            p1 = (xc - wx - hx, yc - wy - hy)
            p2 = (xc + wx - hx, yc + wy - hy)
            p3 = (xc + wx + hx, yc + wy + hy)
            p4 = (xc - wx + hx, yc - wy + hy)
            poly = np.array([conf, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]])
            polygons.append(poly)
        return polygons
        