#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# File: interface.py
# Project: deepsight-back
# Created Date: 一月 08,2020 14:20
# Author: yanlin
# Description: 标记模型分析
# -----
# Last Modified: 2020/01/08 14:20
# Modified By: yanlin
# -----
# Copyright (C) DeepSight - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import torch
import logging
import numpy as np
import torch.nn as nn
from . import _init_paths
from scipy.misc import imread
from django.conf import settings
from torch.autograd import Variable
from model.nms.nms_wrapper import nms
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob
from model.rpn.bbox_transform import clip_boxes
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

logger = logging.getLogger(settings.ENVTYPE)


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def _get_image_blob(im, cfg):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def load_model(model_path, own_data_classes, header='vgg16', cfg_file='cfgs/vgg16.yml', cuda=True, cfg_list=None, mGPUs=False):
  cfg_from_file(cfg_file)
  if not (cfg_list is None):
    cfg_from_list(cfg_list)
    # cfg_from_list(['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20'])
  load_name = model_path

  # initilize the network here.
  if header == 'vgg16':
    fasterRCNN = vgg16(own_data_classes, pretrained=False, class_agnostic=False)
  elif header == 'res101':
    fasterRCNN = resnet(own_data_classes, 101, pretrained=False, class_agnostic=False)
  elif header == 'res50':
    fasterRCNN = resnet(own_data_classes, 50, pretrained=False, class_agnostic=False)
  elif header == 'res152':
    fasterRCNN = resnet(own_data_classes, 152, pretrained=False, class_agnostic=False)
  else:
    logger.debug("network is not defined")
    # pdb.set_trace()

  fasterRCNN.create_architecture()

  if mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  logger.debug("load checkpoint %s" % (load_name))

  if cuda:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  if cuda:
    fasterRCNN.cuda()

  return fasterRCNN

def filter_bndbox(bndbox, ratio=0.2):
  x_min, y_min, x_max, y_max = bndbox[0], bndbox[1], bndbox[2], bndbox[3]
  if x_max-x_min<5:
    return False
  elif y_max-y_min<5:
    return False
  elif float(x_max-x_min)/(y_max-y_min) < ratio:
    return False
  elif float(x_max-x_min)/(y_max-y_min) > 1/ratio:
    return False
  else:
    return True


def inference(input_np, fasterRCNN, own_data_classes, cfg_file='cfgs/vgg16.yml', cuda=True, cfg_list=None):
  cfg_from_file(cfg_file)
  if not (cfg_list is None):
    cfg_from_list(cfg_list)
  cfg.USE_GPU_NMS = cuda
  np.random.seed(cfg.RNG_SEED)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if cuda:
    cfg.CUDA = True
  else:
    cfg.CUDA = False

  fasterRCNN.eval()

  # Load the demo image
  im_in = input_np
  # im_in = np.array(imread(im_file))
  if len(im_in.shape) == 2:
    im_in = im_in[:,:,np.newaxis]
    im_in = np.concatenate((im_in,im_in,im_in), axis=2)
  # rgb -> bgr
  im = im_in[:,:,::-1]

  blobs, im_scales = _get_image_blob(im, cfg)
  assert len(im_scales) == 1, "Only single-image batch implemented"
  im_blob = blobs
  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  im_data_pt = torch.from_numpy(im_blob)
  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
  im_info_pt = torch.from_numpy(im_info_np)

  im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
  im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
  gt_boxes.data.resize_(1, 1, 5).zero_()
  num_boxes.data.resize_(1).zero_()

  rois, cls_prob, bbox_pred, \
  rpn_loss_cls, rpn_loss_box, \
  RCNN_loss_cls, RCNN_loss_bbox, \
  rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

  scores = cls_prob.data
  boxes = rois.data[:, :, 1:5]

  if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred.data
      if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          if cuda:
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
          else:
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
          box_deltas = box_deltas.view(1, -1, 4 * len(own_data_classes))

      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
      pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
  else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  pred_boxes /= im_scales[0]

  scores = scores.squeeze()
  pred_boxes = pred_boxes.squeeze()

  boxes_output = np.empty(shape=[0, 4], dtype=np.uint16)
  gt_classes_output = []
  ishards_output = np.empty(shape=[0], dtype=np.int32)

  thresh = 0.05
  #循环分析每个图片
  for j in xrange(1, len(own_data_classes)):
      inds = torch.nonzero(scores[:,j]>thresh).view(-1)
      # if there is det
      if inds.numel() > 0:
        cls_scores = scores[:,j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
        
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
        for i_box in range(cls_dets.shape[0]):
          if cls_dets[i_box,4]>thresh and filter_bndbox(cls_dets[i_box,:4], ratio=0.2):
            boxes_output = np.append(boxes_output, np.expand_dims(cls_dets[i_box,:4], axis=0), axis=0).astype(np.uint16)
            # ishard is 0 as default.
            ishards_output = np.append(ishards_output, [0], axis=0)
            gt_classes_output.append(own_data_classes[j])

  objs_info = {'boxes': boxes_output,
               'classes_name': gt_classes_output,
               'gt_ishard': ishards_output}

  return objs_info
