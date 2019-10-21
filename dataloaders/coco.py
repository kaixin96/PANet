"""
Load COCO dataset
"""

from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch

from .common import BaseDataset

class COCOSeg(BaseDataset):
    """
    Modified Class for COCO Dataset

    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use (default is 2014 version)
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split + '2014'
        annFile = f'{base_dir}/annotations/instances_{self.split}.json'
        self.coco = COCO(annFile)

        self.ids = self.coco.getImgIds()
        self.transforms = transforms
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch meta data
        id_ = self.ids[idx]
        img_meta = self.coco.loadImgs(id_)[0]
        annIds = self.coco.getAnnIds(imgIds=img_meta['id'])

        # Open Image
        image = Image.open(f"{self._base_dir}/{self.split}/{img_meta['file_name']}")
        if image.mode == 'L':
            image = image.convert('RGB')

        # Process masks
        anns = self.coco.loadAnns(annIds)
        semantic_masks = {}
        for ann in anns:
            catId = ann['category_id']
            mask = self.coco.annToMask(ann)
            if catId in semantic_masks:
                semantic_masks[catId][mask == 1] = catId
            else:
                semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
                semantic_mask[mask == 1] = catId
                semantic_masks[catId] = semantic_mask
        semantic_masks = {catId: Image.fromarray(semantic_mask)
                          for catId, semantic_mask in semantic_masks.items()}

        # No scribble/instance mask
        instance_mask = Image.fromarray(np.zeros_like(semantic_mask, dtype='uint8'))
        scribble_mask = Image.fromarray(np.zeros_like(semantic_mask, dtype='uint8'))

        sample = {'image': image,
                  'label': semantic_masks,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without mean subtraction/normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))

        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample
