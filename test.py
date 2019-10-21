"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex


@ex.automain
def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    transforms = [Resize(size=_config['input_size'])]
    if _config['scribble_dilation'] > 0:
        transforms.append(DilateScribble(size=_config['scribble_dilation']))
    transforms = Compose(transforms)


    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            _log.info(f'### Load data ###')
            dataset = make_data(
                base_dir=_config['path'][data_name]['data_dir'],
                split=_config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=_config['n_steps'] * _config['batch_size'],
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries']
            )
            if _config['dataset'] == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")


            for sample_batched in tqdm.tqdm(testloader):
                if _config['dataset'] == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                suffix = 'scribble' if _config['scribble'] else 'mask'

                if _config['bbox']:
                    support_fg_mask = []
                    support_bg_mask = []
                    for i, way in enumerate(sample_batched['support_mask']):
                        fg_masks = []
                        bg_masks = []
                        for j, shot in enumerate(way):
                            fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                        sample_batched['support_inst'][i][j])
                            fg_masks.append(fg_mask.float().cuda())
                            bg_masks.append(bg_mask.float().cuda())
                        support_fg_mask.append(fg_masks)
                        support_bg_mask.append(bg_masks)
                else:
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                    [query_label.cuda()for query_label in sample_batched['query_labels']], dim=0)

                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)

                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')
