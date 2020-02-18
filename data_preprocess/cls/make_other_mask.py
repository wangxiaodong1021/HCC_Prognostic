import cv2
import numpy as np
import sys
import os
from os import listdir
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import json as js
from multiprocessing import Pool
import time


def find_contours_of_xml(file_path_xml, downsample):
    list_blob = []
    with open(file_path_xml, 'r') as load_f:
        data = js.load(load_f)
    for Lesion in data:
        for value in Lesion['coordinates']:
            list_point = []

            try:
                for coor in value:
                    p_x = coor[0]
                    p_y = coor[1]
                    p_x = p_x / downsample
                    p_y = p_y / downsample
                    list_point.append([p_x, p_y])
                if len(list_point) >= 0:
                    list_blob.append(list_point)
            except:
                continue
    contours = []
    for list_point in list_blob:
        list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                          for point in list_point]
        contour = np.array(list_point_int, dtype=np.int32)
        contours.append(contour)
    return  contours
def make_mask(mask_shape, contours):

    wsi_empty = np.zeros(mask_shape[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    cv2.drawContours(wsi_empty, contours, -1, 255, -1)
    _, contours_o, hierarchy = cv2.findContours(
        wsi_empty,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    wsi_empty = np.zeros(mask_shape[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    cv2.drawContours(wsi_empty, contours_o, -1, 255, -1)
    return  wsi_empty
def other_mask(
        cur_path_xml,
        downsample,
        cur_path_origin,
        cur_name,
        save_location_path
               ):
    contours = find_contours_of_xml(cur_path_xml, downsample)
    print (cur_path_origin)
    wsi_bgr = cv2.imread(cur_path_origin)
    mask_shape = wsi_bgr.shape[0:2]
    tumor_mask_0255 = make_mask(mask_shape, contours)
    print ('==> Saving maks at ' + save_location_path + ' ..')
    file_name = cur_name + '_mask_' + '64.png'
    cur_path_save = save_location_path + file_name
    cv2.imwrite(cur_path_save, tumor_mask_0255)

def save_other_mask(file_path_tif,\
                    file_path_origin,\
                    file_path_xml, \
                    save_location_path,
                    # level
                    ):

    file_name_list_origin = [name for name in listdir(file_path_origin)]
    file_name_list_origin.sort()
    #
    file_name_list_xml = [name for name in listdir(file_path_xml)]
    file_name_list_xml.sort()
    len_origin = len(file_name_list_origin)
    len_xml = len(file_name_list_xml)
    opt_list=[]
    for i, file_name in enumerate(file_name_list_xml):

        print ('==> Finding contours of ' + file_name + ' ..')
        if file_name[-4:] == 'json':
            cur_name = file_name.split(".json")[0]
            file_name_tif = file_path_tif+cur_name+'.svs'
            downsample = 64
            cur_path_xml = file_path_xml + file_name
            print ('==> Making mask of ' + file_name + ' ..')
            cur_path_origin = file_path_origin + cur_name+'_origin_cut_64.png'
            opt_list.append((cur_path_xml,downsample,cur_path_origin,cur_name,save_location_path))
    pool = Pool(5)
    pool.starmap(other_mask, opt_list)
    pool.close()
    pool.join()

if __name__=='__main__':
    file_path_tif = "./data/svs/"
    file_path_ground_truth_xml = "./data/json"
    save_path_jpg = "./data/img/"
    save_path_lab_msk = "./data/lab_mask/other/"
    print("start")
    t1 = time.time()
    save_other_mask(file_path_tif,
                        save_path_jpg, \
                        file_path_ground_truth_xml, \
                        save_path_lab_msk, \
                        )
    t2 = time.time()
    print((t2-t1)/60)

    print('end')