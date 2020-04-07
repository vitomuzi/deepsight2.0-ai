from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
import yaml
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete
# this dataset is only for training data

class own_data(imdb):
    def __init__(self, image_set, dataCfg=None):
        dataCfg = 'cfgs/data_cfg_abnormal.yml' if not dataCfg else dataCfg
        with open(dataCfg, 'r') as f:
            data_cfg = yaml.load(f)
        f.close()
        print(dataCfg)

        db_name = data_cfg['db_name']
        imdb.__init__(self, db_name)
        self._image_set = image_set
        # get db info
        self._classes = data_cfg['classes']
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = data_cfg['image_ext'] # e.g. '.jpg'
        # get image file list in db path
        self._data_path = data_cfg['data_path']
        f_all_itr = (f for f in os.listdir(self._data_path))
        f_itr = filter(lambda f:f.endswith(self._image_ext), sorted(f_all_itr))
        f_itr = map(lambda f:f[:-len(self._image_ext)], f_itr)
        f_list = list(f_itr)
        self._image_index = f_list

        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_name(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_name(self, name):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                                  name + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_data_annotation(name)
                    for name in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_data_annotation(self, name):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, name + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.empty(shape=[0, 4], dtype=np.uint16)
        gt_classes = np.empty(shape=[0], dtype=np.int32)
        overlaps = np.empty(shape=[0, self.num_classes], dtype=np.float32)
        seg_areas = np.empty(shape=[0], dtype=np.float32)
        ishards = np.empty(shape=[0], dtype=np.int32)
        num_HSIL = 0
        num_abnormal = 0

        # Load object bounding boxes into a data frame.
        # clean data
        for obj in objs:
            cls_name = obj.find('name').text.lower().strip()
            if not (cls_name in self._classes):
                if cls_name=='hsil':
                    print("change "+ cls_name + " to HSIL")
                    cls_name = "HSIL"
                else:
                    print("change "+ cls_name + " to abnormal")
                    cls_name = "abnormal"
                ##xiugai 2019 01 20 benlai zhiyou yiju ruxia 
                ##continue
                ##
            if cls_name == "HSIL":
                num_HSIL = num_HSIL + 1
            if cls_name == "abnormal":
                num_abnormal = num_abnormal + 1

            cls = self._class_to_ind[cls_name]
            gt_classes = np.append(gt_classes, cls)

            bbox = obj.find('bndbox')
            # labelImage pixel indexes 1-based
            x1 = float(bbox.find('xmin').text) - 1 if float(bbox.find('xmin').text)>0 else 0
            y1 = float(bbox.find('ymin').text) - 1 if float(bbox.find('ymin').text)>0 else 0
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            if x1 >= x2 or y1>=y2:
                continue

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards = np.append(ishards, difficult)

            boxes = np.append(boxes, np.expand_dims([x1, y1, x2, y2], axis=0), axis=0)

            overlap_vect = np.zeros(self.num_classes, dtype=np.float32)
            overlap_vect[cls] = 1.0
            overlaps = np.append(overlaps, np.expand_dims(overlap_vect, axis=0), axis=0)
            seg_areas = np.append(seg_areas, (x2 - x1 + 1) * (y2 - y1 + 1)) 

        print("HISL has " + str(num_HSIL) + " bndboxes")
        print("abnormal has " + str(num_abnormal) + " bndboxes")
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}