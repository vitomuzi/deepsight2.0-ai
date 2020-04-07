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

import cv2
from scipy.misc import imread


class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(256, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state+input)
        output = self.fc3(state)
        #print(output)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)