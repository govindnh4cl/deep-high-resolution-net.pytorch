# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import logging
import time
import numpy as np

import torch
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

# import pyximport
# pyximport.install()

import _init_paths
from config import cfg
from config import update_config
# from core.loss import JointsMSELoss
# from core.function import validate
# from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger = logging.getLogger()
    # logger, final_output_dir, tb_log_dir = create_logger(
    #     cfg, args.cfg, 'valid')
    #
    # logger.info(pprint.pformat(args))
    # logger.info(cfg)

    # cudnn related setting
    # cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location='cpu'), strict=False)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)

    # # define loss function (criterion) and optimizer
    # criterion = JointsMSELoss(
    #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    # ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=True
    # )

    # # evaluate on validation set
    # validate(cfg, valid_loader, valid_dataset, model, criterion,
    #          final_output_dir, tb_log_dir)


    # switch to evaluate mode
    model.eval()

    count_warm_up = 10
    count_actual = 50
    batch_size = 4

    # ----------------------------------
    in_img = torch.zeros((batch_size, 3, 192, 256), dtype=torch.float32)

    start_time = time.time()
    with torch.no_grad():
        for i in range(count_warm_up):
            outputs = model(in_img)
        print('Time at warm up: {:.0f}ms'.format(1000 * (time.time() - start_time)/(count_warm_up * batch_size)))

    # ----------------------------------
    in_img = torch.zeros((batch_size, 3, 192, 256), dtype=torch.float32)
    start_time = time.time()
    with torch.no_grad():
        for i in range(count_actual):
            outputs = model(in_img)

            # if isinstance(outputs, list):
            #     output = outputs[-1]
            # else:
            #     output = outputs

            # preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
            #
            # all_preds = np.zeros((config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
            # all_boxes = np.zeros(6)
            # all_preds[ :, 0:2] = preds[:, :, 0:2]
            # all_preds[ :, 2:3] = maxvals
            # # double check this all_boxes parts
            # all_boxes[0:2] = c[:, 0:2]
            # all_boxes[2:4] = s[:, 0:2]
            # all_boxes[4] = np.prod(s*200, 1)
            # all_boxes[5] = score

        print('Time: {:.0f}ms'.format(1000 * (time.time() - start_time) / (count_actual * batch_size)))


if __name__ == '__main__':
    main()
