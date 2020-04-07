# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import yaml
import shutil
import urllib
import urllib.error
import urllib.parse
import urllib.request
from PIL import Image
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from . import _init_paths

import cv2
from scipy.misc import imread

#import fasterRCNN._init_paths
#import fasterRCNN.interface as rcnn
#import fasterRCNN.utils.general as ut

from deepsight.service.common.fasterRCNN import _init_paths
from deepsight.service.common.fasterRCNN import interface as rcnn
from deepsight.service.common.fasterRCNN.utils import general as ut


import clsNet._init_paths
import clsNet.interface as clsNet
from clsNet.utilities.utils import save_annt_file

import densenet_step3_v3._init_paths
import densenet_step3_v3.evaluate_new as step3_v3_new

from rnn_model import rnn_single as rnn_net


def load_model(net_path, cuda=True):
    net = rnn_net(128)
    #net = torch.nn.DataParallel(net)

    print("Model {} loading...".format(net_path))
    if cuda:
        checkpoint = torch.load(net_path)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(net_path, map_location=(lambda storage, loc: storage))
        net.load_state_dict(checkpoint)

    if cuda:
        net.cuda()
       # net = torch.nn.DataParallel(net)

    return net


def predict(inputs_new, net, cuda=True):
    state = Variable(net.init_hidden(1)).cuda()
    for s in range(len(inputs_new)):
    	current = inputs_new[s]
    	current_tensor = Variable(torch.from_numpy(current)).cuda()
    	output, state = net(current_tensor, state)
    output = F.softmax(output)
    
    return output

def interface(rnn_inputs_prob, rnn_inputs, length, rnnNet):
    rnn_inputs_return = []	
    input_len = len(rnn_inputs_prob)
    result_zero = 0
    predict_class = 0
    if input_len==0:
        result_zero = 1
        return predict_class, result_zero

    if input_len<length:
        rnn_inputs_prob = rnn_inputs_prob * (int(length/input_len) +1)
        rnn_inputs = rnn_inputs * (int(length/input_len) +1)

    prob_index = np.argsort(rnn_inputs_prob)
    for index in range(1,length+1):
        item_data = rnn_inputs[prob_index[-index]][0]
        rnn_inputs_return.append(item_data)

    predict_result = predict(rnn_inputs_return, rnnNet)
    predict_result = predict_result.data.cpu().numpy()
    print(predict_result)
    predict_class = np.where(abs(predict_result-predict_result.max())<1e-10)[1][0]

    return predict_class, result_zero
