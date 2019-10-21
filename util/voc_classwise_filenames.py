"""
This snippet processes VOC segmentation data
and generates filename list according to the
class labels each image contains.

This snippet will create folders under
"ImageSets/Segmentaion/" with the same
names as the splits. Each folder has 20 txt files
each contains the filenames whose associated image
contains this class label.
"""

import os

import numpy as np
from PIL import Image

# set path
voc_dir = '../../data/Pascal/VOCdevkit/VOC2012/'
seg_dir = os.path.join(voc_dir, 'SegmentationClassAug')
trainaug_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'trainaug.txt')
trainval_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'trainval.txt')
train_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt')
val_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'val.txt')

# list filenames of all segmentation masks
filenames = os.listdir(seg_dir)

# read filenames in different data splits
with open(train_path, 'r') as f:
    train = f.read().splitlines()
with open(val_path, 'r') as f:
    val = f.read().splitlines()
with open(trainval_path, 'r') as f:
    trainval = f.read().splitlines()
with open(trainaug_path, 'r') as f:
    trainaug = f.read().splitlines()
filenames_dic = {'train': train,
                 'val': val,
                 'trainval': trainval,
                 'trainaug': trainaug}

# create a dic to store the classwise filename lists
dic = {'train': {},
       'val': {},
       'trainval': {},
       'trainaug': {}}
for split in dic:
    os.makedirs(os.path.join(voc_dir, 'ImageSets', 'Segmentation', split), exist_ok=True)

# check if each mask contains certain label
for filename in filenames:
    filepath = os.path.join(seg_dir, filename)
    label_set = set(np.unique(np.asarray(Image.open(filepath)))) - set((0, 255)) # exclude 0 and 255
    filename_wo_png = filename.replace('.png', '')
    for label in label_set:
        for split in dic:
            if filename_wo_png in filenames_dic[split]:
                if label in dic[split].keys():
                    dic[split][label].append(filename_wo_png)
                else:
                    dic[split][label] = [filename_wo_png,]

# write the result to file
for split in dic:
    for label in dic[split].keys():
        imageset_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', split,
                                     'class{}.txt'.format(label))
        with open(imageset_path, 'w+') as f:
            for item in dic[split][label]:
                f.write("{}\n".format(item))
