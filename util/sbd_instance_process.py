"""
This snippet processes SBD instance segmentation data
and transform it from .mat to .png. Then transformed
images will be saved in VOC data folder. The name of
the new folder is "SegmentationObjectAug"
"""

import os

from scipy.io import loadmat
from PIL import Image

# set path
voc_dir = '../Pascal/VOCdevkit/VOC2012/'
sbd_dir = '../SBD/'
inst_path = os.path.join(voc_dir, 'SegmentationObject')
inst_aug_path = os.path.join(sbd_dir, 'inst')

# set target dirctory
target_path = os.path.join(voc_dir, 'SegmentationObjectAug')
os.makedirs(target_path, exist_ok=True)

# copy original VOC instance masks
inst_files = os.listdir(inst_path)
for inst_file in inst_files:
    im = Image.open(os.path.join(inst_path, inst_file))
    im.save(os.path.join(target_path, inst_file))
palette = im.getpalette()

# read SBD instance masks and save them
inst_aug_files = os.listdir(inst_aug_path)
for inst_aug_file in inst_aug_files:
    target_file = os.path.join(target_path, inst_aug_file.replace('.mat', '.png'))
    if not os.path.isfile(target_file):
        data = loadmat(os.path.join(inst_aug_path, inst_aug_file))
        im = Image.fromarray(data['GTinst']['Segmentation'][0, 0])
        im.putpalette(palette)
        im.save(target_file)
