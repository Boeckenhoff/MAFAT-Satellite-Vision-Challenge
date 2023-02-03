import numpy as np
import os
from random import sample
import shutil

data_root = './split_640_640'

images_paths = np.asarray(sorted(os.listdir(os.path.join(data_root, "images"))))
labels_paths = np.asarray(sorted(os.listdir(os.path.join(data_root, "annfiles"))))

l = len(images_paths) #length of data 
f = int(0.8 * l)  #number of elements you need
indices = sample(range(l),f)

images_paths_train = images_paths[indices]
labels_paths_train = labels_paths[indices]

images_paths_val = np.delete(images_paths, indices)
labels_paths_val = np.delete(labels_paths, indices)

#create folders with splits

# create folders
os.makedirs(os.path.join("train", "images"))
os.makedirs(os.path.join("train", "labelTxt"))

os.makedirs(os.path.join("val", "images"))
os.makedirs(os.path.join("val", "labelTxt"))

# move files to folders
for img_path, lbl_path in zip(images_paths_train, labels_paths_train):
    src = os.path.join(data_root, "images", img_path)
    dst = os.path.join("train", "images", img_path)
    shutil.copyfile(src, dst)
    
    src = os.path.join(data_root, "annfiles", lbl_path)
    dst = os.path.join("train", "labelTxt", lbl_path)
    shutil.copyfile(src, dst)
    
# move files to folders
for img_path, lbl_path in zip(images_paths_val, labels_paths_val):
    src = os.path.join(data_root, "images", img_path)
    dst = os.path.join("val", "images", img_path)
    shutil.copyfile(src, dst)
    
    src = os.path.join(data_root, "annfiles", lbl_path)
    dst = os.path.join("val", "labelTxt", lbl_path)
    shutil.copyfile(src, dst)