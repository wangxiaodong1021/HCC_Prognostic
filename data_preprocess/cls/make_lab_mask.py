import cv2
import numpy as np

import os
import sys
from os import listdir

from openslide import OpenSlide
import json as js
import struct
import collections
import time
from multiprocessing import Pool
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
np.set_printoptions(threshold=np.nan)
def find_contours_of_xml(file_path_xml, downsample):
    contours = {}
    color = []
    with open(file_path_xml, 'r') as load_f:
        data = js.load(load_f)
    for Lesion in data:
        if Lesion['color'] not in color:
            color.append(Lesion['color'])
    contours = collections.defaultdict(list)
    for cval in color:
        for Lesion in data:
            list_blob = []
            if Lesion['color'] == cval:
                for value in Lesion['coordinates']:
                    list_point = []
                    try:
                        for coor in value:
                            p_x = coor[0]
                            p_y = coor[1]
                            p_x = p_x / downsample
                            p_y = p_y / downsample
                            list_point.append([int(round(p_x)),int(round(p_y))])
                        if len(list_point) >= 0:
                            list_point=np.array(list_point, dtype=np.int32)
                            list_blob.append(list_point)
                    except:
                        continue
                for list_point in list_blob:
                    list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                                      for point in list_point]
                    contour = np.array(list_point_int, dtype=np.int32)

                    contours[cval].append(contour)

    return  contours

def hex2rgb(hex_str):
    str = hex_str.split('#')[-1]
    int_tuple = struct.unpack('BBB', bytes.fromhex(str))
    return tuple([val for val in int_tuple])

def make_tumor_mask(mask_shape, contours):

    wsi_empty = np.zeros(mask_shape[:2])

    wsi_empty = wsi_empty.astype(np.uint8)

    cv2.drawContours(wsi_empty, contours, -1, (255,255,255), -1)
    return wsi_empty

def get_color_list(color,dict):
    color_list = []
    for key ,value in dict.items():
        if key == color:
            color_list.append(key)
    for key ,value in dict.items():
        if key!=color:
            color_list.append(key)
    return color_list

def save_res_mask(
        dict,
        cur_path_origin,
        color_list,
        color,
        save_location_path ):
    print(cur_path_origin)
    wsi_bgr_jpg = cv2.imread(cur_path_origin)
    wsi_jpg_vi = wsi_bgr_jpg.copy()


    print('at make mask 1')
    for val in color_list:
        rgb_color = hex2rgb(val)
        bgr_color = (rgb_color[-1], rgb_color[1], rgb_color[0])
        if val == color:
            cv2.drawContours(wsi_jpg_vi, \
                             dict.get(val),
                             -1,
                             bgr_color,
                             -1)
        else:
            cv2.drawContours(wsi_jpg_vi, \
                             dict.get(val),
                             -1,
                             (255, 255, 255),
                             -1)

    wsi_gray_lv_ = cv2.cvtColor(wsi_jpg_vi, cv2.COLOR_BGR2GRAY)
    ret, wsi_bin_0255_lv_ = cv2.threshold( \
        wsi_gray_lv_,
        0,
        255, \
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_o = np.ones((2, 2), dtype=np.uint8)
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
    mask_shape_lv_ = wsi_gray_lv_.shape
    tissue_mask_lv_ = make_tumor_mask(mask_shape_lv_, contours_tissue_lv_)

    tumor_mask_0255 = make_tumor_mask(mask_shape_lv_, dict.get(color))

    and_mask = cv2.bitwise_and(tissue_mask_lv_, tumor_mask_0255)
    _, contours2, h2 = cv2.findContours(
        and_mask,
        cv2.RETR_CCOMP,
        2)

    hierarchy = np.squeeze(h2)
    if len(hierarchy.shape)==1:
        if (hierarchy[3] != -1 ):
            cv2.drawContours(tumor_mask_0255, contours2, -1,(0, 0, 0), -1)
    else:

        for i in range(len(contours2)):

            if (hierarchy[i][3] != -1 ):
                cv2.drawContours(tumor_mask_0255, contours2, i, (0, 0, 0), -1)


    cv2.imwrite(save_location_path, tumor_mask_0255)



def make_mask(
        dict,
        cur_path_origin,
        save_tumor_mask,
        save_tumor_beside_mask,
        save_necrosis_mask,
        save_filename
):


    for color in dict.keys():
        print('at make color list')
        color_list = get_color_list(color, dict)
        save_location = ''

        if color == '#f5a623':
            save_location = save_tumor_mask
        if color == '#d0021b':
            save_location = save_necrosis_mask
        if color == '#7ed321':
            save_location = save_tumor_beside_mask
        save_mask_and_path = os.path.join(save_location, save_filename)
        print('at make mask')
        save_res_mask(
            dict,
            cur_path_origin,
            color_list,
            color,
            save_mask_and_path)


if __name__ == '__main__':
    print('start')
    t1 = time.time()
    tiff_path = './data/svs/'
    file_path_json = './data/json/'
    file_path_origin ='./data/img/'
    save_tumor_mask = './data/lab_mask/tumor/'
    save_tumor_beside_mask = './data/lab_mask/tumor_beside/'
    save_necrosis_mask = './data/lab_mask/necrosis/'
    level = 3
    opt_list = []
    list_file_name = [f for f in listdir(file_path_json )]
    for i, file_name in enumerate(list_file_name):
        if file_name[-4:] == 'json':
            cur_name = file_name.split('.json')[0]
            cur_tiff_path = os.path.join(tiff_path,cur_name+'.svs')
            print(cur_tiff_path)
            slide = OpenSlide(cur_tiff_path)
            downsample = 64
            cur_path_json =os.path.join(file_path_json,file_name)
            dict = find_contours_of_xml(cur_path_json, downsample)
            cur_path_origin = os.path.join(file_path_origin,cur_name+'_origin_cut_64.png')
            print(cur_path_origin)
            save_filename= cur_name + '_mask_64.png'

            opt_list.append((
                dict,
                cur_path_origin,
                save_tumor_mask,
                save_tumor_beside_mask,
                save_necrosis_mask,
                save_filename))
    pool = Pool(5)
    pool.starmap(make_mask, opt_list)
    pool.close()
    pool.join()

    t2 = time.time()
    print((t2-t1)/60)
    print('end')




