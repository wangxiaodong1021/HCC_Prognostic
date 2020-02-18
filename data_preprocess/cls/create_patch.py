import os
import time
import sys
from multiprocessing import Pool
from openslide import OpenSlide
import numpy as np
from skimage.transform.integral import integral_image, integrate
import cv2
from shutil import copyfile
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
file_path_tif = "./data/svs/"
#tissue mask path
file_path_tis_mask = "./data/tissue_mask/"
#lab mask path
file_path_tumor_mask = "./data/lab_mask/tumor/"
file_path_tumor_beside_mask = "./data/lab_mask/tumor_beside/"

file_path_necrosis_mask = "./data/lab_mask/necrosis/"
file_path_other_mask = './data/lab_mask/other/'

#txt path
file_path_txt_train_tumor = "./data/txt/train_tumor.txt"
file_path_txt_train_tumor_beside = "./data/txt/train_tumor_beside.txt"
file_path_txt_train_fibrous_tissue = "./data/txt/train_fibrous_tissue.txt"
file_path_txt_train_necrosis = "./data/txt/train_necrosis.txt"
file_path_txt_valid_tumor = "./data/txt/valid_tumor.txt"
file_path_txt_valid_tumor_beside = "./data/txt/patch_TCGA/x40/valid_tumor_beside.txt"
file_path_txt_valid_fibrous_tissue = "./data/txt/valid_fibrous_tissue.txt"
file_path_txt_valid_necrosis = "./data/txt/valid_necrosis.txt"

#json_path
file_path_json = "./data/json/"
#patch path
patch_path = "./data/patch_cls/"


patch_size = 768
patch_size_lv = 768
num_process = 5
patch_level = 0
mask_downsample= 64
stride = 1024
# divide training set & valid set
proportion=0.7
#tumor_threshold
threshold=0.7
#normal_threshold
normal_threshold=0.1
#sample number for each slide
number=2000

num=5000

#divide tiff to train&valid
def divide(file_path_tif,file_path_normal_mask,file_path_tumor_mask,proportion,file_path_json):

    list_all=[]
    list_normal=[]
    list_tumor=[]
    list_train=[]
    list_valid=[]

    for file in os.listdir(file_path_tif):
        list_all.append(file[:-5])
    # random.shuffle(list_all)

    for file in os.listdir(file_path_normal_mask):
        list_normal.append(file[:-12])
    for file in os.listdir(file_path_tumor_mask):
        list_tumor.append(file[:-12])

    for i in range(math.floor(len(list_normal)*proportion)):
        list_train.append(list_normal[i])
    list_valid = list(set(list_normal).difference(set(list_train)))
    list_all = list(set(list_all).difference(set(list_normal)))

    list_tumor = list(set(list_tumor) - (set(list_tumor) & set(list_normal)))
    for i in range(math.floor(len(list_tumor)*proportion)):
        list_train.append(list_tumor[i])
    for i in range(math.floor(len(list_tumor)*proportion),len(list_tumor)):
        list_valid.append(list_tumor[i])

    list_all = list(set(list_all).difference(set(list_tumor)))
    for i in range(math.floor(len(list_all)*proportion)):
        list_train.append(list_all[i])
    for i in range(math.floor(len(list_all)*proportion),len(list_all)):
        list_valid.append(list_all[i])


    with open(file_path_json, "w+")as file:
        file.write("list_train=")
        file.write(str(list_train))
        file.write("\n")
        file.write("list_valid=")
        file.write(str(list_valid))
        file.write("\n")

    return(list_train,list_valid)


def create_txt_sequence(file_path_tif,file_path_tis_mask,file_path_mask,list,\
                     file_path_txt,kind,patch_size,mask_downsample,stride):

    txt = open(file_path_txt,'a+')
    for file in os.listdir(file_path_json):
        if file[-3:] == 'txt':
            continue
        if "TCGA-2Y-A9H1" in file:
            continue
        file_name=file[:-5]
        lab_mask_name = file_path_mask + file_name + '_mask_' + str(mask_downsample) + '.png'
        if (file_name in list) and (os.path.exists(lab_mask_name)):
            print(lab_mask_name)
            tissue_mask_name = file_path_tis_mask + file_name + '_tissue_mask_' + str(mask_downsample) + '.png'
            print(tissue_mask_name)
            tissue_mask = cv2.imread(tissue_mask_name, 0)
            integral_image_tissue = integral_image(tissue_mask.T / 255)
            # Make integral image of slide
            lab_mask = cv2.imread(lab_mask_name, 0)
            integral_image_lab = integral_image(lab_mask.T / 255)
            print(file_path_tif + file_name + '.svs')
            slide = OpenSlide(file_path_tif + file_name + '.svs')
            slide_w_lv_0, slide_h_lv_0 = slide.dimensions
            slide_w_downsample = slide_w_lv_0 / mask_downsample
            slide_h_downsample = slide_h_lv_0 / mask_downsample
            size_patch_lv_k = int(patch_size / mask_downsample)  # patch在第mask_level层上映射的大小

            _, contours_lab, _ = cv2.findContours(lab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            p_left = []
            p_right = []
            p_bottom = []
            p_top = []
            # Extract random patches on tissue region
            for contour in contours_lab:
                coordinates = (np.squeeze(contour)).T
                coords_x = coordinates[0]
                coords_y = coordinates[1]
                # Bounding box vertex
                p_left.append(np.min(coords_x))
                p_right.append(np.max(coords_x))
                p_top.append(np.min(coords_y))
                p_bottom.append(np.max(coords_y))

            p_x_left = min(p_left)
            p_x_right = max(p_right)
            p_y_top = min(p_top)
            p_y_bottom = max(p_bottom)

            stride_lv=int(stride/mask_downsample)
            print(stride_lv)
            list_type = []
            for contour in contours_lab:
                for x in range(p_x_left, p_x_right, stride_lv):
                    for y in range(p_y_top, p_y_bottom, stride_lv):
                        x_lv = int(x + size_patch_lv_k / 2)
                        y_lv = int(y + size_patch_lv_k / 2)
                        if (y + size_patch_lv_k> slide_h_downsample) or \
                                (x + size_patch_lv_k > slide_w_downsample):
                            continue
                        tissue_integral = integrate(integral_image_tissue, \
                                                    (x, y), \
                                                    (x + size_patch_lv_k - 1,y + size_patch_lv_k - 1))
                        tissue_ratio = tissue_integral / (size_patch_lv_k ** 2)
                        lab_integral = integrate(integral_image_lab, \
                                                 (x, y), \
                                                 (x + size_patch_lv_k - 1,
                                                  y + size_patch_lv_k - 1)
                                                 )
                        lab_ratio = lab_integral / (size_patch_lv_k ** 2)
                        if tissue_ratio < 0.8 or lab_ratio < 0.9:
                            continue
                        list_type.append([x, y])


            for item in list_type:
                x = item[0]
                y = item[1]


                patch_x_lv_0 = str((round(int(x + size_patch_lv_k / 2) * mask_downsample)))
                patch_y_lv_0 = str((round(int(y + size_patch_lv_k / 2) * mask_downsample)))

                txt.writelines([file_name, ',', patch_x_lv_0, ',', patch_y_lv_0, ',', kind, '\n'])


    txt.close()


def create_other_txt_sequence(file_path_tif,file_path_tis_mask,file_path_mask,list,\
                     file_path_txt,kind,patch_size,mask_downsample,stride):

    txt = open(file_path_txt,'a+')
    for file in os.listdir(file_path_tif):
        if "TCGA-2Y-A9H1" in file:
            continue
        file_name=file[:-4]
        lab_mask_name = file_path_mask + file_name + '_mask_' + str(mask_downsample) + '.png'
        if (file_name in list) and (os.path.exists(lab_mask_name)):
            tissue_mask_name = file_path_tis_mask + file_name + '_tissue_mask_' + str(mask_downsample) + '.png'
            tissue_mask = cv2.imread(tissue_mask_name, 0)
            integral_image_tissue = integral_image(tissue_mask.T / 255)
            # Make integral image of slide
            lab_mask = cv2.imread(lab_mask_name, 0)
            integral_image_lab = integral_image(lab_mask.T / 255)
            size_patch_lv_k = int(patch_size / mask_downsample)  # patch在第mask_level层上映射的大小
            print(size_patch_lv_k)
            slide = OpenSlide(file_path_tif + file)
            slide_w_lv_0, slide_h_lv_0 = slide.dimensions
            slide_w = int(slide_w_lv_0 / mask_downsample)
            slide_h = int(slide_h_lv_0 / mask_downsample)
            _, contours_lab, _ = cv2.findContours(lab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            _, contours_tissue, _ = cv2.findContours(tissue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            p_left = []
            p_right = []
            p_bottom = []
            p_top = []
            # Extract random patches on tissue region
            for contour in contours_tissue:
                coordinates = (np.squeeze(contour)).T
                coords_x = coordinates[0]
                coords_y = coordinates[1]
                # Bounding box vertex
                p_left.append(np.min(coords_x))
                p_right.append(np.max(coords_x))
                p_top.append(np.min(coords_y))
                p_bottom.append(np.max(coords_y))

            p_x_left = min(p_left)
            p_x_right = max(p_right)
            p_y_top = min(p_top)
            p_y_bottom = max(p_bottom)

            stride_lv=int(stride/mask_downsample)
            print( p_x_left,p_x_right,p_y_top,p_y_bottom)

            for x in range(p_x_left, p_x_right, stride_lv):
                for y in range(p_y_top, p_y_bottom, stride_lv):

                    if (x + size_patch_lv_k - 1 >= slide_w) or (y + size_patch_lv_k - 1 >= slide_h):
                        continue
                    tissue_integral = integrate(integral_image_tissue, \
                                                (x, y), \
                                                (x + size_patch_lv_k - 1,
                                                 y + size_patch_lv_k - 1)
                                                )
                    tissue_ratio = tissue_integral / (size_patch_lv_k ** 2)
                    lab_integral = integrate(integral_image_lab, \
                                             (x, y), \
                                             (x + size_patch_lv_k - 1,
                                              y + size_patch_lv_k - 1)
                                             )
                    lab_ratio = lab_integral / (size_patch_lv_k ** 2)
                    if tissue_ratio > 0.8 and lab_ratio < 0.2:
                        x_lv = int(x + size_patch_lv_k / 2)
                        y_lv = int(y + size_patch_lv_k / 2)
                        patch_x_lv_0 = str((round(x_lv * mask_downsample)))
                        patch_y_lv_0 = str((round(y_lv * mask_downsample)))


                        txt.writelines([file_name,',',patch_x_lv_0,',',patch_y_lv_0,',',kind,'\n'])


    txt.close()


# create patch
def process(opts):
    i, pid, x_center, y_center, file_path_tif, path_patch, patch_size, patch_level = opts
    x = int(int(float(x_center)) - 1024 / 2)
    y = int(int(float(y_center)) - 1024 / 2)
    wsi_path = os.path.join(file_path_tif + pid + '.svs')
    slide = OpenSlide(wsi_path)
    img = slide.read_region(
        (x, y), 0,
        (1024, 1024)).convert('RGB')
    if not os.path.exists(os.path.join(path_patch, pid + "_" + str(i) + '.png')):

        print(os.path.join(path_patch, pid + "_" + str(i) + '.png'))
        img.save(os.path.join(path_patch, pid + "_" + str(i) + '.png'))



# multiprocess
def run(file_path_tif,file_path_txt,patch_path,kind,num_process,patch_size,patch_level):
    par_filename = file_path_txt.split('/')[-1]
    par_filename = par_filename.split('.')[0]
    par_dir = os.path.join(patch_path, kind)
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)
    sub_dir = os.path.join(par_dir, par_filename)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    copyfile(file_path_txt, os.path.join(sub_dir, "list.txt"))
    opts_list = []
    list = []
    infile = open(file_path_txt)
    for i, line in enumerate(infile):
        print(line)
        print(line.strip('\n').split(','))
        pid, x_center, y_center, _ = line.strip('\n').split(',')
        list.append([pid, x_center, y_center, kind])
    count = len(list)
    print(count)
    infile.close()
    for folder_num in range(0, math.ceil(count / num)):
        path_patch = os.path.join(sub_dir, str(folder_num))
        if os.path.exists(path_patch) == False:
            os.mkdir(path_patch)
        #
        for i in range(folder_num * num, folder_num * num + num):
            try:
                pid = list[i][0]
                x_center = list[i][1]
                y_center = list[i][2]
                opts_list.append((i, pid, x_center, y_center, file_path_tif, path_patch, patch_size, patch_level))
            except:
                break
    pool = Pool(processes=num_process)
    pool.map(process, opts_list)



if __name__ == '__main__':

    list_train = ['TCGA-CC-A5UE-01Z-00-DX1.11245DF8-BDD4-435F-9329-A4F63C24EC88',
                  'TCGA-2Y-A9H9-01Z-00-DX1.4C1CCB4D-6011-4275-B562-28BFFA5F0C7F',
                  'TCGA-2Y-A9H3-01Z-00-DX1.813C14D8-DBBC-43AF-9E3A-9EFC1D1AFC98',
                  'TCGA-BC-A3KG-01Z-00-DX1.FC63E3C7-C7AF-45F0-8DD4-7968B8425C09',
                  'TCGA-CC-A9FS-01Z-00-DX1.DBE01DEE-CEF0-4224-BB06-DDD687000770',
                  'TCGA-CC-A3MA-01Z-00-DX1.96F9CA78-2690-4FAF-859A-D3D9CCBA98F1',
                  'TCGA-2Y-A9H1-01Z-00-DX1.FE4D124E-AB92-4083-8D09-F025B0C637EB',
                  'TCGA-CC-A5UC-01Z-00-DX1.5844CBD9-61B4-47D3-A5D9-C283926BF710',
                  'TCGA-CC-A8HS-01Z-00-DX1.267BA6C7-41E1-4E93-AB80-ECB6D082F49E',
                  'TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018',
                  'TCGA-CC-A7IH-01Z-00-DX1.3905C6E9-5AD8-4135-8215-9D41DFD61DFA',
                  'TCGA-BC-A110-01Z-00-DX1.E46C2B24-A159-4970-82B8-8F6CD60FDFB0',
                  'TCGA-BC-A10T-01Z-00-DX1.2F634D24-C571-49E1-ACAE-B19F8AFF71D3',
                  'TCGA-CC-A7IJ-01Z-00-DX1.525F3A40-46CC-4455-9473-F99BFA2D3572',
                  'TCGA-CC-5264-01Z-00-DX1.11E7D28A-D9CD-4CDF-898C-7CA6C6940A4A',
                  'TCGA-CC-A3MC-01Z-00-DX1.680CCC54-8159-42A9-8B66-FEC9228EB8BC',
                  'TCGA-BC-A10X-01Z-00-DX1.5F143963-21D2-4067-ABEB-08B2AC02EEB3',
                  'TCGA-2Y-A9H2-01Z-00-DX1.D1B4968D-8AAD-4F12-8447-6FC752E8DDE4',
                  'TCGA-CC-5258-01Z-00-DX1.5183124B-973C-4583-B9E0-FB7015E2DC1B',
                  'TCGA-BC-A69H-01Z-00-DX1.22CA7EA4-B8EC-444C-90AA-33095CC53585',
                  'TCGA-BC-A10U-01Z-00-DX1.6EF0CB11-CB54-4469-B5C8-2C21B733A138',
                  'TCGA-CC-5259-01Z-00-DX1.D18B2F62-0CC0-4665-90E3-7DB6817F2FF8',
                  ]
    list_valid = ['TCGA-2Y-A9H0-01Z-00-DX1.E1725D1E-B135-47B3-820B-BEE1FC4EDCF0',
                  'TCGA-2Y-A9GX-01Z-00-DX1.DC3E1EAB-686A-44F9-91AF-0C6975DE029E',
                  'TCGA-BC-A10R-01Z-00-DX1.6A5135A9-04DB-42B5-AD16-5DC224CCAA2B',
                  'TCGA-CC-A7IF-01Z-00-DX1.F5D74385-ABD3-466D-9DCD-2A762140F633',
                  'TCGA-CC-A7IE-01Z-00-DX1.1286E91C-EB64-4CEB-878F-436D0E70F275',
                  'TCGA-2Y-A9GW-01Z-00-DX1.71805205-933D-4D72-A4A2-586DC5490D76',
                  'TCGA-CC-A9FU-01Z-00-DX1.DC790527-1973-4609-AB9A-271246B36918']

    # #
    print('Creating txt!')
    time_now = time.time()

    create_txt_sequence(file_path_tif, file_path_tis_mask,file_path_tumor_beside_mask,list_train,\
               file_path_txt_train_tumor_beside, 'tumor_beside', patch_size, mask_downsample, stride)
    create_txt_sequence(file_path_tif, file_path_tis_mask,file_path_tumor_beside_mask, list_valid,\
               file_path_txt_valid_tumor_beside, 'tumor_beside', patch_size, mask_downsample, stride)

    create_txt_sequence(file_path_tif, file_path_tis_mask,file_path_tumor_mask,list_train,\
               file_path_txt_train_tumor, 'tumor', patch_size, mask_downsample, stride)
    create_txt_sequence(file_path_tif, file_path_tis_mask,file_path_tumor_mask, list_valid,\
               file_path_txt_valid_tumor, 'tumor', patch_size, mask_downsample, stride)
    # #
    create_txt_sequence(file_path_tif,file_path_tis_mask, file_path_necrosis_mask, list_train,\
               file_path_txt_train_necrosis, 'necrosis', patch_size, mask_downsample, stride)
    create_txt_sequence(file_path_tif,file_path_tis_mask,file_path_necrosis_mask,list_valid ,\
               file_path_txt_valid_necrosis, 'necrosis', patch_size, mask_downsample, stride)

    create_other_txt_sequence(file_path_tif, file_path_tis_mask, file_path_other_mask, list_train, \
                              file_path_txt_train_fibrous_tissue, 'fibrous_tissue', patch_size, mask_downsample, stride)
    create_other_txt_sequence(file_path_tif, file_path_tis_mask, file_path_other_mask, list_valid, \
                              file_path_txt_valid_fibrous_tissue, 'fibrous_tissue', patch_size, mask_downsample, stride)
    time_spent = (time.time() - time_now) / 60
    print('Creating txt for %f min!' % time_spent)

    #
    print('Making patch!')
    time_now = time.time()

    run(file_path_tif, file_path_txt_train_tumor_beside, patch_path, "train", num_process, patch_size_lv, patch_level)
    run(file_path_tif, file_path_txt_train_fibrous_tissue, patch_path, "train", num_process, patch_size_lv, patch_level)
    run(file_path_tif, file_path_txt_train_necrosis, patch_path, "train", num_process, patch_size_lv, patch_level)

    run(file_path_tif, file_path_txt_valid_tumor, patch_path, "valid", num_process, patch_size_lv, patch_level)
    run(file_path_tif, file_path_txt_train_tumor, patch_path, "train", num_process, patch_size_lv, patch_level)

    run(file_path_tif, file_path_txt_valid_tumor_beside, patch_path, "valid", num_process, patch_size_lv, patch_level)
    run(file_path_tif, file_path_txt_valid_fibrous_tissue, patch_path, "valid", num_process, patch_size_lv, patch_level)
    run(file_path_tif, file_path_txt_valid_necrosis, patch_path, "valid", num_process, patch_size_lv, patch_level)

    time_spent = (time.time() - time_now) / 60
    print('Making patch  for %f min!' % time_spent)
