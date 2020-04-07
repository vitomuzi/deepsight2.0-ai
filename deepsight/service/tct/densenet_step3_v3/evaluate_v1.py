import argparse
import os
import numpy as np
import torch

from sklearn import metrics
from torch.autograd import Variable

import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn

import random
import argparse
import math
import yaml

from loader import load_data
from modelmodel import MRNet

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser

def run_model(model, loader, train=False, optimizer=None, gpu = True):

    if gpu:
        model = torch.nn.DataParallel(model).cuda()

    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch in loader:

        vol, label = batch
        labels.append(label)
        if gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)
        label = Variable(label)

        logit = model.forward(vol)

        loss = criterion(logit, label)
        total_loss += loss.data[0]

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
       # label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        #labels.append(label_npy)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, diagnosis, use_gpu):
    train_loader, valid_loader, test_loader = load_data(diagnosis, use_gpu)

    model = MRNet()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    loss, auc, preds, labels = run_model(model, loader)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

def load_model(net_path, own_data_classes, cuda=True):
    net = MRNet()
   # net = torch.nn.DataParallel(net)
    
    print("Model {} loading...".format(net_path))
    if cuda:
        checkpoint = torch.load(net_path)
        print('load path')
        net.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(net_path, map_location=(lambda storage, loc: storage))
        net.load_state_dict(checkpoint)


    if cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)

    return net

def interface(patches, probality_s, net, own_data_classes, cuda=True):
    ind_to_class = dict(zip(range(len(own_data_classes)), own_data_classes))
    result_flag = 0
    if len(probality_s)==0:
        result_flag = 1
        return 0, 1.0, result_flag

    mean = [0.485, 0.45, 0,406]
    std = [0.229, 0.224, 0.225]

    if len(probality_s)<5:
        probality_s = probality_s * (math.floor(5/len(probality_s) + 1 ) + 1 )
        patches =  patches * (math.floor(5/len(patches) + 1 ) + 1 )
    
    seq = np.zeros((5,224,224,3), dtype=np.float32)
    pro_index = np.argsort(probality_s)
    for img_index in range(1,6):
        seq_index = img_index - 1
        item_data = patches[pro_index[-img_index]]
        seq[seq_index,:,:,0] = (item_data[:,:,0]/255. - mean[0])/std[0]
        seq[seq_index,:,:,1] = (item_data[:,:,1]/255. - mean[1])/std[0]
        seq[seq_index,:,:,2] = (item_data[:,:,2]/255. - mean[2])/std[2]

    try:
        predict_result = predict(seq, net, cuda = cuda)
    except Exception as e:
        raise e
    predict_result = predict_result.data.cpu().numpy()

    return predict_result, 1.0 , result_flag

def predict(seq, net, cuda=True):
    X = torch.from_numpy(np.rollaxis(seq,3,1)).float()

    if cuda:
        X = Variable(X, volatile=True).cuda()
    else:
        X = Variable(X, volatile=True)

    y = net(X)

    if not cuda:
        y = y.cpu()

    return y


def _get_img(box, img, class_name):
    xmin, ymin, xmax, ymax = box
    x_center = (xmin+xmax)/2
    y_center = (ymin+ymax)/2
    x_center = 112 if x_center<112 else x_center
    y_center = 112 if y_center<112 else y_center
    x_center = 912 if x_center>912 else x_center
    y_center = 912 if y_center>912 else y_center

    xmin = x_center - 112
    xmax = x_center + 112
    ymin = y_center - 112
    ymax = y_center + 112

    box = [xmin, ymin, xmax, ymax]

    return [box], [class_name], [np.asarray(img.crop(box))]

def get_testImg(img_np, annt_info_refined, own_data_classes):
    boxes = []
    classes = []
    patch = []
    total_probality = []
    _class_to_ind = dict(zip(own_data_classes, range(len(own_data_classes))))

    if len(annt_info_refined['classes_name']) == 0:
        # return empty list if annt file is empty
        return boxes, classes, patch, total_probality

    img = Image.fromarray(img_np)

    for idx, box_annt in enumerate(annt_info_refined['boxes']):
        box_item, box_class_item, patch_item = _get_img(box_annt, img, _class_to_ind[annt_info_refined['classes_name'][idx]])
        boxes.extend(box_item)
        classes.extend(box_class_item)
        patch.extend(patch_item)
        total_probality.append(annt_info_refined['gt_probablity'][idx])

    return boxes, classes, patch, total_probality


if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)
