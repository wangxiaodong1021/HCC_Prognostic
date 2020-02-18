import cv2
import numpy as np
from openslide import OpenSlide
import sys
from multiprocessing import Pool
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
def save_slide_cutting_multiple(
        file_path,
        save_location,
        multiple

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
        if downsample > multiple:
            continue
        elif downsample < multiple:
            downsample = multiple/downsample
            w = int(w_lv_/downsample)
            h = int(h_lv_/downsample)
            img = cv2.resize(wsi_bgr_lv_, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(save_location, img)
            break
        else:
            img = wsi_bgr_lv_
            cv2.imwrite(save_location, img)
            break

def save_slide_cutting(file_path, save_location, level):
    slide = OpenSlide(file_path)
    print('==> saving slide_lv_' + str(level) + ' at ' + save_location)
    x_lv_, y_lv_ = 0, 0
    w_lv_, h_lv_ = slide.level_dimensions[level]
    try:
        wsi_pil_lv_ = slide.read_region((0, 0), level, \
                                        (w_lv_, h_lv_))

        wsi_ary_lv_ = np.array(wsi_pil_lv_)
        wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(save_location, wsi_bgr_lv_)
    except:
        print(file_path)


if __name__=='__main__':
    file_path_tif = "./data/svs/"
    save_path_jpg = "./data/img/"

    print("start")

    multiple = 64
    for root, dirs, files in os.walk(file_path_tif):
        for file_name in files:
            if '.svs' in file_name:
                cur_file_path = os.path.join(root,file_name)
                print(cur_file_path)
                file_name = file_name.replace('.svs', '')

                file_name = file_name + '_origin_cut_64.png'
                cur_save_loca = save_path_jpg + file_name
                print('save at ' + cur_save_loca)
                save_slide_cutting_multiple(cur_file_path, cur_save_loca, multiple)

    print("end")

