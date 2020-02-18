import os
import sys
import time
import numpy as np
from openslide import OpenSlide
from multiprocessing import Pool
import cv2
import csv
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
path_list = "./data/npy/"
file_path_txt = "./data/txt/x40.txt"
path_patch = "./data/patch_prognostic/x40/"
file_path_tif = "./data/svs/"



stride = 5
num_process = 5
patch_size = 256
patch_level = 0

def get_coordinates(np_path,file_name):
    np_file = np.load(np_path)
    list_tumor = []

    list_tumor_new = []


    txt = open(file_path_txt, 'a+')
    [rows, cols] = np_file.shape
    for i in range(0, rows-5, stride):
        for j in range(0, cols-5, stride):
            # 癌症
            if int(np_file[i, j]) == 2:
                flag1 = 0
                for index_w in range(i, i + 5):
                    for index_h in range(j, j + 5):
                        if np_file[index_w, index_h] == 2:
                            flag1 = flag1 + 1
                if flag1 == 5 * 5:
                    list_tumor.append([i,j])

            else:
                continue

    if len(list_tumor) > 150:
        list_index = random.sample(range(len(list_tumor)), 150)
        for number in list_index:
            list_tumor_new.append(list_tumor[number])
        list_tumor = list_tumor_new





    for item in list_tumor:
        x = item[0]
        y = item[1]
        patch_x_lv_0=str(x*256)
        patch_y_lv_0=str(y*256)

        txt.writelines([file_name, ',', patch_x_lv_0, ',', patch_y_lv_0,',tumor', '\n'])



    txt.close()


def cut(file_path_txt,file_path_tif,patch_path,patch_size,patch_level):
    opts_list = []


    infile = open(file_path_txt)
    for i, line in enumerate(infile):

        pid, x, y, kind = line.strip('\n').split(',')
        dir = pid.split("/")[-1]
        dir_split=dir.split("-")[:3]
        pid_par = dir_split[0] + '-' + dir_split[1] + '-' + dir_split[2]
        pid_dir = os.path.join(patch_path, pid_par)

        if not os.path.exists(pid_dir):
            os.mkdir(pid_dir)
        class_dir = os.path.join(pid_dir, kind)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        opts_list.append((i, pid, x, y, file_path_tif,patch_path, patch_size, patch_level,class_dir))
    count = len(opts_list)
    print(count)
    infile.close()
    pool = Pool(processes=num_process)
    pool.map(process, opts_list)

def process(opts):
    i, pid, x, y, file_path_tif, patch_path, patch_size, patch_level,class_dir= opts
    dir = pid.split("/")[-1]
    dir_split = dir.split("-")[:3]
    dir_name = dir_split[0] + '-' + dir_split[1] + '-' + dir_split[2]


    x=int(float(x))
    y=int(float(y))
    wsi_path = os.path.join(file_path_tif, dir + '.svs')

    slide = OpenSlide(wsi_path)
    img = slide.read_region(
            (x, y), patch_level,
            (patch_size, patch_size))
    wsi_ary_lv_ = np.array(img)
    img = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(os.path.join(class_dir, dir_name + "_" + str(i) + '.png'), img)


if __name__=='__main__':

    for root, dirs, files in os.walk(path_list):
        for file in files:
            file_name = file.split(".npy")[0]
            np_path = os.path.join(root,file)
            file_path = root.split('/')[-1]
            file_path = file_path+'/'+file_name
            pid_split = file_name.split('-')[:3]
            pid = pid_split[0] + '-' + pid_split[1] + '-' + pid_split[2]
            get_coordinates(np_path, file_path)



    print('Making patch!')
    time_now = time.time()

    cut(file_path_txt, file_path_tif, path_patch, patch_size, patch_level)
    time_spent = (time.time() - time_now) / 60
    print('Making patch  for %f min!' % time_spent)
