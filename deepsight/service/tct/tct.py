# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import shutil
import urllib
import urllib.error
import urllib.parse
import urllib.request
import requests

import numpy as np
import cv2
from scipy.misc import imread
import random

from . import  _init_paths

from deepsight.service.common.fasterRCNN import _init_paths
from deepsight.service.common.fasterRCNN import interface as rcnn
from deepsight.service.common.fasterRCNN import load
from clsNet import interface as clsNet

from PIL import Image
import uuid

import tcthandle

import densenet_step3_v3._init_paths
import densenet_step3_v3.evaluate_new as step3_v3

import Inflammation._init_paths
import Inflammation.interface as inflam_interface
from django.conf import settings

import case_part._init_paths
import case_part.interface as case_interface
from tqdm import tqdm
from configs.config import *
import logging

from deepsight.service.common import download_tctfile


#if __name__ == "__main__":
def tct(number,url):
    logging.debug('\t----------------------------------------------------------------------')
    #parser = argparse.ArgumentParser(description='inference on local')
    #parser.add_argument('--num_fold', help='the num of fold for test', default=1, type=int)
    #parser.add_argument('--index', help='choose a part of f_itr', default=0, type=int)
    #parser.add_argument('--case_folder', help='folder of the input cases', type = str)
    #parser.add_argument('--annt_info_outdir', help='output anntation dir', type=str)
    #parser.add_argument('--case_result_outdir', help='case diagnosis output dir', type=str)
    #args = parser.parse_args()
    # load models
    #class_det = ('__background__', '2')
    download_tctfile.down_load_tct(url)
    class_det = ('__background__', 'abnormal')
    # faster RCNN
    header='res101'
    #model_path = 'fasterRCNN/data/models/res101/own_data/faster_rcnn_sc_ah128_bs16_1_20_726.pth'
    #model_path = '/opt/test/common/fasterRCNN/data/models/res101/own_data/faster_rcnn_abnormal_ah128_bs16_1_20_1470.pth'
    #cfg_file = '/opt/test/common/fasterRCNN/cfgs/{}.yml'.format(header)
    cfg_list = ['ANCHOR_SCALES', '[16, 32, 64, 128]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    logging.debug('start RCNN load')
    #fasterRCNN = rcnn.load_model(model_path, class_det, header=header, cfg_file=cfg_file, cuda=True, cfg_list=cfg_list,mGPUs=True)
    #fasterRCNN = rcnn.load_model(model_path, class_det, header=header, cfg_file=cfg_file, cuda=True, cfg_list=cfg_list)
    fasterRCNN ,cfg_file = load.faster_rcnnload(class_det, header, cfg_list)
    logging.debug('faster RCNN loaded')
    print('faster RCNN loaded')

    # cls net
    #class_cls = ('normal', 'abnormal', 'H-SIL', 'Candida', 'fungi', 'Herpesviruses')
    class_cls = ('normal', 'abnormal')
    
    net_path = os.path.join(settings.BASE_DIR,'deepsight/service/tct/clsNet/data/models/refine_bs16_lr0.01_alldb/CP91.pth')

    print(net_path)

    logging.debug('start denseNet load')
    net = clsNet.load_model(net_path, class_cls, cuda=True)
    print('DenseNet loaded')
    logging.debug('DenseNet loaded')


    #step3Net  10 
    step3Net_path = os.path.join(settings.BASE_DIR,'deepsight/service/tct/densenet_step3_v3/data/models/time02/epoch10.pth')
    logging.debug('start step3Net load')
    step3Net = step3_v3.load_model(step3Net_path, class_cls, cuda=True)
    print('step3Net loaded')
    logging.debug('step3Net loaded')

    #case_part
    rnnNet_path = os.path.join(settings.BASE_DIR,'deepsight/service/tct/case_part/case_model/bs4_lr0.001/19_rnn_checkpoint_best.pth')
    logging.debug('start caseNet load')
    rnnNet = case_interface.load_model(rnnNet_path)
    print('caseNet loaded ')
    logging.debug('caseNet loaded ')
 
    #fungi------part
    fungi_det =  ('__background__', 'fungi')
    fungi_path = os.path.join(settings.BASE_DIR,'deepsight/service/tct/Fungi/models/faster_rcnn_own_1_20_1579.pth')
    logging.debug('start fungiNet load')
    fungi_net = rcnn.load_model(fungi_path, fungi_det, header=header, cfg_file=cfg_file, cuda=True, cfg_list=cfg_list)
    print('fungiNet loaded')
    logging.debug('fungiNet loaded')

    #inflammation
    infla_cls = ['normal','mild','moderate']
    infla_path = os.path.join(settings.BASE_DIR,'deepsight/service/tct/Inflammation/checkpoints/CP36.pth')
    logging.debug('start infla_net load')
    infla_net = inflam_interface.load_model(infla_path, infla_cls, cuda=True)
    print('infla_net loaded')
    logging.debug('infla_net loaded')

    # 拉取所有尚未分析过的片子
    case_folder = settings.TCT_IMAGE+"/"+number+"/torch"
    #slides = _get_case_dir(case_folder)
    # print(slides)
    #args.index
    #args.num_fold


    #输出预测的标注结果文件夹
    #annt_info_outdir = '/root/zhoum/CC/clsNet/data/own_data/output/fastrcnn_clsnet/annt_info_outdir_10img_001'
    annt_info_outdir = settings.TCT_CELL_IMAGE +"/"+number

    #输出的病人诊断结果文件夹
    
    case_result_outdir = settings.TCT_PATIENT_IMAGE

    # 对每张片子进行处理
    try:
        print(case_folder)
        result_json = tcthandle.handleSlide(case_folder, fasterRCNN, clsNet, step3Net, class_det, class_cls, cfg_file, cfg_list,annt_info_outdir, case_result_outdir, fungi_net,fungi_det,infla_net , rnnNet, net, check_num = 200)
    except Exception as e:
        logging.exception("Exception occurred")
    # 完成
    print("-------------Done-------------"+str(case_folder))
    logging.debug("\t-------------Done-------------"+str(case_folder))
