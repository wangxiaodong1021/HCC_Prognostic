import cv2
import numpy as np

from os import listdir
import os
import sys
from openslide import OpenSlide
from multiprocessing import Pool
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
def make_tumor_mask(mask_shape, contours):
    print('at make tumor mask')
    t1 = time.time()
    wsi_empty = np.zeros(mask_shape[:2])
    print(wsi_empty.size)
    wsi_empty = wsi_empty.astype(np.uint8)
    print('draw')
    cv2.drawContours(wsi_empty, contours, -1, (255,255,255), -1)
    t2 = time.time()
    print((t2-t1)/60)
    return wsi_empty

def make_mask(
        img,
        save_location
):

    wsi_gray_lv_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('gray')
    ret, wsi_bin_0255_lv_ = cv2.threshold( \
        wsi_gray_lv_,
        0,
        255, \
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print('otsu')
    kernel_o = np.ones((2, 2), dtype=np.uint8)
    kernel_c = np.ones((2, 2), dtype=np.uint8)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_CLOSE, \
        kernel_c)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_OPEN, \
        kernel_o)
    _, contours_tissue_lv_, hierarchy = \
        cv2.findContours( \
            wsi_bin_0255_lv_, \
            cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE)
    mask_shape_lv_ = img.shape
    print(mask_shape_lv_)
    tissue_mask_lv_ = make_tumor_mask(mask_shape_lv_, contours_tissue_lv_)
    print('at save')
    cv2.imwrite(save_location,tissue_mask_lv_)

def choose_level(
        file_path
):
    slide = OpenSlide(file_path)
    level_cnt = slide.level_count
    for level in reversed(range(level_cnt)):
        downsample = slide.level_downsamples[level]
        w_lv_, h_lv_ = slide.level_dimensions[level]
        wsi_pil_lv_ = slide.read_region(
            (0, 0),
            level,
            (w_lv_, h_lv_))
        wsi_ary_lv_ = np.array(wsi_pil_lv_)
        wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
        downsample = round(downsample)
        print(downsample)
        if downsample > 64:
            continue
        elif downsample < 64:
            downsample = 64/downsample
            w = int(w_lv_/downsample)
            h = int(h_lv_/downsample)
            img = cv2.resize(wsi_bgr_lv_, (w, h), interpolation=cv2.INTER_LINEAR)
            return img
        else:
            img = wsi_bgr_lv_
            return img

def make_level_mask(
        level,
        file_path,
        save_location
):
    slide = OpenSlide(file_path)
    w_lv_, h_lv_ = slide.level_dimensions[level]
    print(w_lv_,h_lv_)
    wsi_pil_lv_ = slide.read_region((0, 0),
                                    level, \
                                    (w_lv_, h_lv_))

    img = np.array(wsi_pil_lv_)
    wsi_gray_lv_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('gray')
    ret, wsi_bin_0255_lv_ = cv2.threshold( \
        wsi_gray_lv_,
        0,
        255, \
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print('otsu')
    kernel_o = np.ones((4, 4), dtype=np.uint8)
    kernel_c = np.ones((4, 4), dtype=np.uint8)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_CLOSE, \
        kernel_c)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_OPEN, \
        kernel_o)
    _, contours_tissue_lv_, hierarchy = \
        cv2.findContours( \
            wsi_bin_0255_lv_, \
            cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE)
    mask_shape_lv_ = img.shape
    print(mask_shape_lv_)
    tissue_mask_lv_ = make_tumor_mask(mask_shape_lv_, contours_tissue_lv_)
    print('at save')
    cv2.imwrite(save_location,tissue_mask_lv_)
if __name__ == '__main__':
    tiff_path = './data/svs/'
    print('start')
    t1 = time.time()
    save_location = './data/tissue_mask/'
    print('start')
    list_file_name = [f for f in listdir(tiff_path)]
    opt_list = []
    for i, file_name in enumerate(list_file_name):
        if file_name.split('.')[-1] == 'svs':
            cur_name = file_name.split('.svs')[0]
            file_path = tiff_path+file_name
            img = choose_level(file_path)
            save_path =save_location+cur_name+'_tissue_mask_'+'64.png'
            opt_list.append((
                img,
                save_path))
    pool = Pool(5)
    pool.starmap(make_mask, opt_list)
    pool.close()
    pool.join()
    t2=time.time()
    print((t2-t1)/60)





