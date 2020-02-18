import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
import xlrd
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torchvision import transforms
import random
import cv2

random.seed(0)


class ImageDataset(nn.Module):
    def __init__(self, data_path, normalize=True, way="train", factor='40', val=False, type_key='tumor',
                 ExperimentWay='prognosis'):
        super(ImageDataset, self).__init__()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])
        self._crop_size = 224
        self._way = way
        self.factor = factor
        self.val = val
        self.data_path = data_path + '/x' + factor + '/' + self._way
        print(self.data_path)
        files = os.listdir(self.data_path)
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._normalize = normalize
        self.img_pth = []
        self.Observed = []
        self.Survival = []
        self.ExperimentWay = ExperimentWay
        self.read_csv()
        self.less = ['10-6953', '10-7446', '10-7121', '11-0344', '11-7421', '11-3471', '10-7114', '10-2933', '11-0896',
                     '10-4588', '12-4098', '10-2900', '10-7122', '12-30762', '11-16005', '12-35977', '12-20275',
                     '12-27791', '12-24058', '11-19506', '12-05611']
        self.problem_file = ['12-6735', '12-3383', '12-7816', 'TCGA-5C-AAPD', 'TCGA-BW-A5NO', 'TCGA-DD-A3A8',
                             'TCGA-BC-A3KF', 'TCGA-WQ-A9G7', 'TCGA-G3-AAV4', 'TCGA-T1-A6J8', 'TCGA-ZP-A9D1',
                             'TCGA-EP-A2KC', 'TCGA-DD-AAC8', 'TCGA-RC-A6M5', 'TCGA-DD-A11B', 'TCGA-FV-A4ZQ',
                             'TCGA-FV-A496', 'TCGA-RC-A6M6', 'TCGA-DD-AACK', 'TCGA-DD-A4NR', 'TCGA-ED-A7PX',
                             'TCGA-ED-A7PZ', 'TCGA-ED-A97K', 'TCGA-FV-A495']
        self.sample_num = {'4': 8, '10': 16, '40': 16}
        self.type_key = type_key
        for file in files:
            if file not in self.problem_file and file not in self.less:
                if file not in self.person_dict.keys():
                    continue
                self.img_pth.append(os.path.join(self.data_path, file))
                self.Observed.append(self.person_dict[file][1])
                self.Survival.append(self.person_dict[file][0])

    def read_csv(self):  # 0 alive; 1 dead
        data = pd.read_csv('./csv/tcga.csv')
        Survival = data.OS.values
        Observed = data.vital_status.values
        Id =list(data.PID)
        self.person_dict = dict(zip(Id, zip(Survival, Observed)))
    def Id_process(self, Id):
        Id_copy = []
        for i in Id:
            tmp = i.split('-')[1][0]
            if tmp == '0':
                Id_copy.append(i.split('-')[0][-2:] + '-' + i.split('-')[1][1:])
            else:
                Id_copy.append(i.split('-')[0][-2:] + '-' + i.split('-')[1])
        return Id_copy

    def _preprocess(self, im):
        if self._way == "train":
            im = self._color_jitter(im)
            if np.random.rand() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            num_rotate = np.random.randint(0, 4)
            im = im.rotate(90 * num_rotate)
            # crop
            im = np.array(im)
            img_h, img_w, _ = im.shape
            pad_h = max(self._crop_size - img_h, 0)
            pad_w = max(self._crop_size - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(im, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(255.0, 255.0, 255.0))
            else:
                img_pad = im
            h_off = random.randint(0, img_h - self._crop_size)
            w_off = random.randint(0, img_w - self._crop_size)
            im = np.asarray(img_pad[h_off: h_off + self._crop_size, w_off: w_off + self._crop_size])
        im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        if self._normalize:
            im = (im - 128.0) / 128.0
        return im

    def __len__(self):
        return len(self.img_pth)

    def __getitem__(self, idx):
        T = self.Survival[idx]
        O = self.Observed[idx]
        pth = []
        type_path = os.path.join(self.img_pth[idx] + '/' + self.type_key)

        files = os.listdir(type_path)

        for file in files:
            img_size = os.path.getsize(type_path + '/' + file)
            img_size = round((img_size / float(1024)), 2)
            if self.type_key == 'fibrous_tissue':
                if img_size <= 90:
                    files.remove(file)
            else:
                if img_size <= 105:
                    files.remove(file)
        if self.val:
            sample_num = 16
        else:
            sample_num = self.sample_num[self.factor]
        if len(files) < sample_num:

            imgs = files
        else:
            imgs = random.sample(files, sample_num)
        count = len(imgs)
        Imgs = []
        for img in imgs:
            pth.append(type_path + '/' + img)
            Img = Image.open(type_path + '/' + img)
            Img = self._preprocess(Img)
            Imgs.append(Img)
        Imgs = np.array(Imgs)
        return Imgs, T, O, pth, count
