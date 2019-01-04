# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
from pathlib import Path
import numpy as np
import imageio

def bb_hw(a): 
    return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])

def hw_bb(bb): 
    return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])

def crop_img(bbox_dict, root_path, tgt_path):
    for img_name in bbox_dict.keys():
        print(img_name)
        img_path = root_path/img_name
        # im = open_image(img_path)
        im = imageio.imread(img_path)

        im_shape = im.shape
        ratio_arr = [im_shape[1] / sz, im_shape[0] / sz, im_shape[1] / sz, im_shape[0] / sz]
        # print(ratio_arr)
        # im_resize = transform.resize(im, (sz, sz, 3), mode='constant')
        # ax = show_img(im)

        bb = bbox_dict[img_name]
        bb_pred = [min(sz, max(0, o)) for o in bb[0]]
        b = bb_hw(bb_pred)
        b_ratio = np.multiply(b, ratio_arr)
        b_ratio = [int(v) for v in b_ratio]
        # print(b)
        # print(b_ratio)
        b_crop = hw_bb(b_ratio)
        if len(im_shape) == 3:
            im_crop = im[b_crop[0]:b_crop[2], b_crop[1]:b_crop[3],:]
        else:
            im_crop = im[b_crop[0]:b_crop[2], b_crop[1]:b_crop[3]]
        # show_img(im_crop)
        imageio.imwrite(Path(tgt_path)/img_name, im_crop)

PATH = Path('./data')
TRAIN = 'train'
TRAIN_PATH = PATH/TRAIN
TEST = 'test'
TEST_PATH = PATH/TEST

TGT_TRAIN_PATH = Path('./data/train_crop')
TGT_TRAIN_PATH.mkdir(parents=True, exist_ok=True)
TGT_TEST_PATH = Path('./data/test_crop')
TGT_TEST_PATH.mkdir(parents=True, exist_ok=True)

sz=224

with open('./data/test_bbox.pk', 'rb') as f:
    test_bbox_dict = pickle.load(f)

crop_img(test_bbox_dict, TEST_PATH, TGT_TEST_PATH)

with open('./data/train_bbox.pk', 'rb') as f:
    train_bbox_dict = pickle.load(f)

crop_img(train_bbox_dict, TRAIN_PATH, TGT_TRAIN_PATH)
