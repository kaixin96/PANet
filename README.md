# PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment

This repo contains code for our ICCV 2019 paper [PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment](https://arxiv.org/abs/1908.06391).

### Dependencies

* Python 3.6 +
* PyTorch 1.0.1
* torchvision 0.2.1
* NumPy, SciPy, PIL
* pycocotools
* sacred 0.7.5
* tqdm 4.32.2

### Data Preprocessing

Preprocess PASCAL VOC data (with SBD augmentation) before running the few-shot segmentation code. VOC data folder looks like

```
VOC2012
├── Annotations
├── ImageSets
├── JPEGImages
├── SegmentationClass
├── SegmentationClassAug
└── SegmentationObject
```

1. Generate `png` format instance annotation data for SBD augmentation.

    Use `PANet/util/sbd_instance_process.py` to process SBD instance segmentation data and transform it from .mat to .png. Save the output `png`s into `VOC2012/SegmentationObjectAug/`. You may need to configure the path before running this snippet.

2. Generates filename list according to the object class labels each image contains.

    Use `PANet/util/voc_classwise_filenames.py` to process VOC segmentation data and and generates filename list according to the object class labels each image contains. Check the snippet source for details. You may need to configure the path before running this snippet.

3. Generates scribble annotations.

    The scribble annotations are available [here](https://drive.google.com/file/d/1H6x5QPj4T4dFkq9qChYt0TNEenEBFvE_/view?usp=sharing). Download it and put it under `VOC2012` folder. The annotations are generated using `PANet/util/scribbles.py`.

After preprocessing, the data folder should look like

```
VOC2012
├── Annotations
├── ImageSets
├── JPEGImages
├── ScribbleAugAuto
├── SegmentationClass
├── SegmentationClassAug
├── SegmentationObject
└── SegmentationObjectAug
```


### Usage

1. Download the ImageNet-pretrained weights of VGG16 network from `torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and put it under `pretrained_model` folder.

2. Change configuration via `config.py`, then train the model using `python train.py` or test the model using `python test.py`. You can use `sacred` features, e.g. `python train.py with gpu_id=2`.

### Citation
Please consider citing our paper if the project helps your research. BibTeX reference is as follows.
```
@article{wang2019panet,
  title={PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment},
  author={Wang, Kaixin and Liew, Jun Hao and Zou, Yingtian and Zhou, Daquan and Feng, Jiashi},
  journal={arXiv preprint arXiv:1908.06391},
  year={2019}
}
```