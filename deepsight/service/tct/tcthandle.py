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

import _init_paths

import numpy as np
import cv2
from scipy.misc import imread
import random

#import fasterRCNN._init_paths
#import fasterRCNN.interface as rcnn
#import fasterRCNN.utils.general as ut

from deepsight.service.common.fasterRCNN import interface as rcnn
#from common.fasterRCNN import interface as rcnn
import deepsight.service.common.fasterRCNN


#import clsNet._init_paths
#import clsNet.interface as clsNet
from clsNet import _init_paths
from clsNet import interface as clsNet
from clsNet.utilities.utils import save_annt_file

from PIL import Image
import uuid

#import densenet_step3_v3._init_paths
#import densenet_step3_v3.evaluate_new as step3_v3
from densenet_step3_v3 import _init_paths
from densenet_step3_v3 import evaluate_new as step3_v3

#import Inflammation._init_paths
#import Inflammation.interface as inflam_interface
from Inflammation import _init_paths
from Inflammation import interface as inflam_interface

#import case_part._init_paths
#import case_part.interface as case_interface
from case_part import _init_paths
from case_part import interface as case_interface

from tqdm import tqdm
from configs.config import *
import logging

def _get_case_dir(case_folder):
  case_dir = filter(os.path.isdir, [os.path.join(case_folder, f) for f in os.listdir(case_folder)])
  return [os.path.join(case_folder, d) for d in case_dir]

#产生新的xml文件名称
def _get_targetXML_file_path(out_dir,slicename,filename):
    slice_dir,slice_only_name = os.path.split(slicename)
    slice_dir,slice_only_name = os.path.split(slice_dir)
    f_name,fe_name = os.path.splitext(filename) 
    #return os.path.join(out_dir,f_name+'.xml')
    return os.path.join(out_dir,slice_only_name,f_name+'.xml') , slice_only_name

def get_file_list(img_dir, img_suffix='.jpg'):
    f_all_itr={}
    for i_name in os.listdir(img_dir):
        m = os.path.join(img_dir,i_name)
        if os.path.isfile(m):
            f_all_itr[i_name]=os.path.getsize(m)
    sort_result = sorted(f_all_itr.items(),key = lambda asd:asd[1],reverse=True)

    f_itr = []
    for item in sort_result:
        if item[0].endswith(img_suffix):
            f_itr.append(item[0])
    f_itr = map(lambda f:f[:-len(img_suffix)], f_itr)

    return f_itr

# 片子slide即case所对应的文件夹
def fetchImages(slide):
    file_itr = get_file_list(img_dir=slide, img_suffix='.jpg')

    return [os.path.join(slide, f+'.jpg') for f in file_itr]


def uploadResult(no,contentArray,analysis_post_dict):
    mapContent = {
        "no" : no,
        "content"   : contentArray,
        "inflammation":analysis_post_dict['inflammation'],
        # 细胞量 [">5000", "<5000"]
        "cellAmount":analysis_post_dict['cellAmount'],
        # 细胞项目->鳞状上皮 ["有", "未见"]
        "squamousEpithelium":analysis_post_dict['squamousEpithelium'],
        # fixed 细胞项目->颈管细胞 ["有", "未见"]
        "endocervicalCell":analysis_post_dict['endocervicalCell'],
        # fixed 细胞项目->化生细胞 ["有", "未见"]
        "metaplasticCell":analysis_post_dict['metaplasticCell'],
        #病毒感染 ["HPV感染", "未见", "疱疹病毒感染"]
        "virusInfection":analysis_post_dict['virusInfection'],
        #微生物项目 ["阴道滴虫", "真菌，形态学上符合念珠菌属", "菌群变化，提示细菌性阴道病", "细菌，形态学上符合放线菌属", "细胞变化，符合单纯疱疹病毒感染", "无"]
        "microorganismProject":analysis_post_dict['microorganismProject'],
        #初筛鳞状上皮细胞分析 ["未见上皮内病变及恶性病变（NILM）", "不确定或者低级别病变", "高级别鳞状上皮病变"]
        "squamousEpitheliumResult":analysis_post_dict['squamousEpitheliumResult']
    }
    #调用deepsight后端tps报告接口
    headers = {"Content-Type":"application/json"}
    upload_api = BACKEND_IP+'api/slide/torch/informationEntry'
    mapcontent = json.dumps(mapContent)
    #logging.debug('Call '+BACKEND_IP+'api/slide/torch/informationEntry interface')
    #result = requests.post(upload_api,mapcontent,headers = headers)
    #logging.debug(result)
    #print(mapcontent)
    return mapcontent


# convert analysis array to upload content
def analysis_to_post(num_boxes_array, infla_tag, fungi_tag):
    print(num_boxes_array)
    post_dict = {
        # 炎症  ["无", "轻度", "中度", "重度"]
        "inflammation":"无",
        # 细胞量 [">5000", "<5000"]
        "cellAmount":"<5000",
        #细胞项目->鳞状上皮 ["有", "未见"] 2019-9-5 update 未见 to 有
        "squamousEpithelium":"有",
        # fixed 细胞项目->颈管细胞 ["有", "未见"]
        "endocervicalCell":"未见",
        # fixed 细胞项目->化生细胞 ["有", "未见"]
        "metaplasticCell":"未见",
        #病毒感染 ["HPV感染", "未见", "疱疹病毒感染"]
        "virusInfection":"未见",
        #微生物项目 ["阴道滴虫", "真菌，形态学上符合念珠菌属", "菌群变化，提示细菌性阴道病", "细菌，形态学上符合放线菌属", "细胞变化，符合单纯疱疹病毒感染", "无"]
        "microorganismProject":"无",
        #初筛鳞状上皮细胞分析 ["未见上皮内病变及恶性病变（NILM）", "不确定或者低级别病变", "高级别鳞状上皮病变"]
        "squamousEpitheliumResult":""
    }
    # class_cls = ('normal', 'abnormal', 'H-SIL', 'Candida', 'fungi', 'Herpesviruses')
    num_cell = np.sum(num_boxes_array)
    #post_dict['cellAmount'] = '<5000' if num_cell<5000 else '>5000'
    post_dict['cellAmount'] = '>5000'
    post_dict['squamousEpithelium'] = '有'
    #post_dict['squamousEpithelium'] = '有' if fungi_tag > 0 else '未见'
    #if num_boxes_array[5] > 0:
    #    post_dict['virusInfection'] = '疱疹病毒感染'
    # 判断逻辑需要调整
    #if num_boxes_array[4] > 0:
    #    post_dict['microorganismProject'] = '细菌，形态学上符合放线菌属'
    #if num_boxes_array[3] > 0:
    #    post_dict['microorganismProject'] = '真菌，形态学上符合念珠菌属'
    # 判断逻辑需要优化
    #if num_boxes_array[2] > 0:
    #    post_dict['squamousEpitheliumResult'] = '高级别鳞状上皮病变'
    #elif num_boxes_array[1] > 0:
    if num_boxes_array[1] > 0:
        post_dict['squamousEpitheliumResult'] = '不确定或者低级别病变'
    else:
        post_dict['squamousEpitheliumResult'] = '未见上皮内病变及恶性病变（NILM）'
    #炎症
    infla_list = ["轻度", "轻度", "中度", "重度"]
    post_dict['inflammation'] = infla_list[infla_tag]
    #fungi 
    fungi_list = ["无","真菌"]
    post_dict['microorganismProject'] = fungi_list[fungi_tag]

    return post_dict



# convert xml info to required dict
#def annot_to_post(file_name, annt_info, type_dict, max_index, class_cls):
def annot_to_post(file_name, annt_info, type_dict, max_index, class_cls,probs):
    key_list = [key for key in type_dict]
    content_dict = {}
    content_value = {}
    # print(key_list)
    for key in key_list:
        content_dict[key] = []
        content_value[key] = []

    # # load all boxes
    # for idx, cls_name in enumerate(annt_info['classes_name']):
    #     if cls_name in key_list:
    #         content_dict[cls_name].append(annt_info['boxes'][idx].tolist())

    # choose a box
    for cls_index in range(len(class_cls)):
        try:
            if not(max_index[cls_index] is None):
                #content_dict[class_cls[cls_index]].append(annt_info['boxes'][max_index[cls_index]])
                content_dict[class_cls[cls_index]].append(annt_info['boxes'][max_index[cls_index]].tolist())
                content_value[class_cls[cls_index]].append(probs[max_index[cls_index]].tolist())
        except IndexError as e:
            print(e)
            print(max_index)

    contentArray = {}

    for key in content_dict:
        type_content_dict = {}
        type_content_dict['filename'] = file_name
        type_content_dict['type'] = type_dict[key]
        type_content_dict['rects'] = content_dict[key]
        type_content_dict['confidenceValue'] = content_value[key]
        contentArray[key] = type_content_dict
            
    return contentArray

# 处理一张片子
def handleSlide(slide, fasterRCNN, clsNet,step3Net, class_det, class_cls, cfg_file, cfg_list,annt_info_outdir, case_result_outdir,fungi_net, fungi_det, infla_net, rnnNet, net, check_num = 200):
    # 根据一个片子的编号，去下载用于分析的图片。
    # (因为目前并不是每个片子都对应有影像图，所以先统一用sample这个影像图作为所有片子的影像图)
    # 后面这个地方应该传 slideNo
    file_list = fetchImages(slide)
    #最大考察的视野图个数
    if len(file_list)>check_num:
        file_list = file_list[:check_num:1]
    # the uploaded content is sampled from the contentArray
    # type_dict = {cls name of net output: upload cls name}
    #type_dict = {'normal':'SC', 'abnormal':'LSIL', 'H-SIL':'H_SIL', 'Candida':'NILM', 'fungi':'NILM', 'Herpesviruses':'NILM'}
    type_dict = {'normal':'SC', 'abnormal':'abnormal'}
    key_list = [key for key in type_dict]
    # init contentArray
    contentArray = {}
    for key in key_list:
        contentArray[key] = []
    # record the num of boxes for each array
    num_boxes_array = np.zeros(len(class_cls))
    num_frequence_class_array = np.zeros((len(class_cls),15))

    num_all_img = 0
    num_abnormal_img = 0
    num_fungi_img = 0
    num_inflammation_img = [0,0,0]
    patches = []
    probality_s = []

    rnn_inputs = []
    rnn_inputs_prob = []
    print('11')
    for image_path in tqdm(file_list):
        print('1')
        im_in = np.array(imread(image_path))
        [dir_name, file_name]=os.path.split(image_path)
        #Abnormal cell detection 第一步，异常细胞检测
        #logging.debug('First step, start abnormal cell detection')
        objs_info = rcnn.inference(im_in, fasterRCNN, class_det, cfg_file='fasterRCNN/cfgs/res101.yml', cuda=True, cfg_list=cfg_list)
        #refine 进行去除假阳性操作
        #logging.debug('Second step, start refine operation')
        annt_info_refined, proba_all, max_index, num_boxes_array_img ,num_frequence_class_array_img = clsNet.interface(im_in, objs_info, net, class_cls, cuda=True)
        
 
        #annt_info_outfile_path_XML = _get_targetXML_file_path(annt_info_outdir,slide,file_name)
        annt_info_outfile_path_XML ,slice_only_name = _get_targetXML_file_path(annt_info_outdir,slide,file_name)
        if os.path.exists(os.path.join(annt_info_outdir, slice_only_name)) == False:
            os.mkdir(os.path.join(annt_info_outdir, slice_only_name))
        #print(annt_info_outfile_path_XML)
        save_annt_file(annt_info_outfile_path_XML,annt_info_refined)

        num_boxes_array = num_boxes_array + num_boxes_array_img
        num_frequence_class_array = num_frequence_class_array + num_frequence_class_array_img
        #print(num_boxes_array)

        #统计每张视野图是否是异常视野
        num_all_img += 1
        
        #step3 视野图判断
        #logging.debug('Third step, Judge the field of view')
        boxes, classes, patch, total_probality = step3_v3.get_testImg(im_in, annt_info_refined, class_cls)
        #predict_class,  predict_result,  result_flag = step3_v3.interface(patch, total_probality, step3Net, class_cls)
        predict_class,predict_result,result_flag,rnn_input = step3_v3.interface_new(patch, total_probality, step3Net, class_cls)
        if predict_class==1:
            num_abnormal_img += 1 

        #get rnninput
        if result_flag==0:
            rnn_inputs.append(rnn_input)
            if predict_class==1:
                rnn_inputs_prob.append(predict_result)
            else:
                rnn_inputs_prob.append(1-predict_result)
        
        #fungi----part
        fungi_info = rcnn.inference(im_in, fungi_net, fungi_det, cfg_file=cfg_file, cuda=True, cfg_list=cfg_list)
        #for class_name in objs_info['classes_name']:
        for class_name in fungi_info['classes_name']:
            if class_name == 'fungi':
                num_fungi_img += 1
                break
        
        #inflammation-----part
        infla_result = inflam_interface.interface(im_in, infla_net, cuda=True)
        num_inflammation_img[infla_result] += 1 

        contentArray_img = annot_to_post(file_name, objs_info, type_dict, max_index, class_cls, proba_all)
        for key in key_list:
            #if len(contentArray_img[key]["rects"])>0:
            #    contentArray[key].append(contentArray_img[key])
            if len(contentArray_img[key]['rects'])>0 and contentArray_img[key]['rects'][0][0]-50>0 and contentArray_img[key]['rects'][0][1]-50>0 and contentArray_img[key]['rects'][0][2]-50>0 and contentArray_img[key]['rects'][0][3]-50>0:
                if len(contentArray_img[key]['rects'])>0 and contentArray_img[key]['rects'][0][0]+50<1024 and contentArray_img[key]['rects'][0][1]+50<1024 and contentArray_img[key]['rects'][0][2]+50<1024 and contentArray_img[key]['rects'][0][3]+50<1024:
                    if len(contentArray_img[key]["confidenceValue"])>0:
                        contentArray[key].append(contentArray_img[key])

    contentArray_upload_tmp = []
    for key in key_list:
        random.shuffle(contentArray[key])
        contentArray_upload_tmp.extend(contentArray[key][:10])
    slice_dir,slice_only_name = os.path.split(slide)
    slice_dir,slice_only_name = os.path.split(slice_dir)
    #logging.debug('start uploading thumbnail.....')
    for index,app_id in enumerate(contentArray_upload_tmp):
        tmp_path = slide + '/'+app_id['filename']
        #logging.info('Uploading thumbnail.....')
        if(len(app_id['rects']) == 1):
            x = (app_id['rects'][0][0]+app_id['rects'][0][2])//2
            y = (app_id['rects'][0][1]+app_id['rects'][0][3])//2
            #tmp_tuple =tuple()
            tmp_tuple = []
            tmp_tuple.append(x-50)
            tmp_tuple.append(y-50)
            tmp_tuple.append(x+50)
            tmp_tuple.append(y+50)
            tmp_tuple = tuple(tmp_tuple)
            pil_im = Image.open(tmp_path)
            region = pil_im.crop(tmp_tuple)
            root_path = '/opt/results/case'
            save_image = root_path+'/'+str(uuid.uuid1())+'.jpg'
            region.save(save_image)
            ret_dir = slice_only_name+'/case'
            #正式环境文件夹
            #ret_dir = name+tmp_dir
            files = {'file':open(save_image, 'rb')}
            options={'output':'json','path':ret_dir,'scene':'default','code':''}
            # 上传到文件服务器
            result_tmp = requests.post(HOST,files=files,data=options,verify=False)
            test_result = eval(result_tmp.text)
            #logging.info(test_result)
            #获取每个上传图片的路径的访问地址
            test_result = test_result['url']
            #logging.debug('upload picture is: '+test_result)
            #把uploadpath 添加到tps报告中
            contentArray_upload_tmp[index].update({'uploadPath':test_result})
            #移除上传过的图片
            os.remove(save_image)

    #contentArray_upload = []
    #for key in key_list:
        #random.shuffle(contentArray[key])
        #contentArray_upload.extend(contentArray[key][:10])
    
    ##logging.debug('End upload thumbnail ') 
    #num_boxes_array[2] presents fungi; 1 is has; 0 is donot have
    if num_fungi_img>0:
        fungi_tag = 1
    else:
        fungi_tag = 0
    #logging.debug('fungi is (1 is has; 0 is donot): '+str(fungi_tag))

    #num_boxes_array[3] presents inflammation; 2 is Moderate; 1 is Mild ; 0 is donot have
    if num_inflammation_img[2] > 100:
        infla_tag = 2
    elif num_inflammation_img[1] > 100:
        infla_tag = 1
    else:
        infla_tag = 0
    #logging.debug('Degree of inflammation is: '+str(infla_tag))
    #print(num_boxes_array)
    analysis_post_dict = analysis_to_post(num_boxes_array, infla_tag, fungi_tag)

    contentArray_upload = json.dumps(contentArray_upload_tmp, separators=(',', ':'))
    #rnn样本诊断 case_predict_class 为1 样本结果为abnormal ,0为normal
    case_predict_class, _ = case_interface.interface(rnn_inputs_prob, rnn_inputs, 10, rnnNet)

    print(num_boxes_array)
    print(num_frequence_class_array)
    print(num_abnormal_img , num_all_img)
    print('case_part_pred', case_predict_class)
    print('fungi ',num_fungi_img)
    print('inflammation  ', num_inflammation_img)
    #logging.debug(num_boxes_array)
    #logging.debug(num_frequence_class_array)
    #logging.debug(str(num_abnormal_img) + ' '+ str(num_all_img))
    #logging.debug('case_part_pred'+ str(case_predict_class))
    #logging.debug('fungi '+str(num_fungi_img))
    #logging.debug('inflammation  '+ str(num_inflammation_img))
    
    # older version method
    #final_label = ['normal']
    #if num_abnormal_img>20:
    #    final_label[0] = 'abnormal'

    # v0.9.2 version method
    final_label = ['normal']
    if case_predict_class == 1:
        final_label[0] = 'abnormal'
    
    slice_dir,slice_only_name = os.path.split(slide)
    slice_dir,slice_only_name = os.path.split(slice_dir)
    #print(slice_only_name,'    ', final_label)
    file_total = open(os.path.join(case_result_outdir,'total'+'.txt'), mode = 'a')
    file_total.write('\n')
    file_total.write(slice_only_name+'   '+str(num_abnormal_img)+'   '+str(case_predict_class) )
    file_total.close()
    np.savetxt( os.path.join(case_result_outdir,slice_only_name+'.txt') ,final_label,fmt='%s',delimiter=',')
    #logging.debug('Sample results is: '+str(final_label[0]) +'case_predict_class: '+ str(case_predict_class))
    #squamousEpitheliumResult
    analysis_post_dict['squamousEpitheliumResult'] = final_label[0]
    #调用上传结果的函数，把分析数据给到后端
    #logging.debug('uploadREsult function  to calling  back-end interface')
    result_json = uploadResult(slice_only_name,contentArray_upload,analysis_post_dict)
    
    return result_json
    
    #params = {'aiNumber': slice_only_name, 'squamousEpitheliumResult': final_label[0]}
    #logging.debug('Call '+str(DEEPSITHT_IP_API)+' interface')
    #调用deepsight后端修改接口
    #r = requests.get(DEEPSITHT_IP_API, params)
    # 返回结果
    #ret = r.content
    #logging.debug(ret)
    # json转字典
    #data = json.loads(ret)
    #status = str(data['state'])
    #shutil.rmtree(DELETE_PATH+slice_only_name)

    #with open('./data.txt','a+') as f:
    #    f.write(status)    

    # analysis_post_dict = analysis_to_post(num_boxes_array)
   


 
