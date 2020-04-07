import argparse
import json
import numpy as np
import os
import torch
import yaml

import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from evaluate import run_model
from loader import CustomDataset
from modelmodel import MRNet

def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def train(model, config, check_point_dir = 'data/models', use_gpu = True):
    epochs = config['Train']['num_epoch']
    lr_decay_period = config['Train']['lr_decay_period']
    db_reload_period = config['Train']['db_reload_period']
    batch_size = config['Train']['batchsize']

    train_loader = torch.utils.data.DataLoader(CustomDataset(config, 'train', shuffle=True), batch_size=config['Train']['batchsize'], shuffle=False)
    valid_loader = torch.utils.data.DataLoader(CustomDataset(config, 'val', shuffle=True), shuffle=False)


    if use_gpu:
        model = model.cuda()

    lr = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.85, weight_decay=0.0005)
    if epochs%lr_decay_period == 0:
            adjust_learning_rate(optimizer)

    best_val_loss = float('inf')

    start_time = datetime.now()

    for epoch in range(epochs):
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            #file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch+1}'
            file_name = 'epoch' + str(epoch) + '.pth'
            save_path = check_point_dir + '/' + file_name
            torch.save(model.state_dict(), save_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    return parser

if __name__ == '__main__':

    cfg_path = 'cfgs/train.yaml'

    with open(cfg_path, 'r') as f:
        data_cfg = yaml.load(f)
    f.close()
    print(cfg_path)

    gpu = True
    net = MRNet()

    train(net, data_cfg, check_point_dir='data/models/time02', use_gpu=gpu)
