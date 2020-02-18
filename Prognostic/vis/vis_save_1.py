import torch
import os
import numpy as np
import matplotlib
from pyod.models.knn import KNN
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import cv2
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='Predicting survival time')
parser.add_argument('--data_path', '-d_p', default='tcga/data/patch_prognostic/', type=str,
                    help='data path')
parser.add_argument('--factor', '-factor', default='10', type=str, help='valid way, 40 10 or combinate')
parser.add_argument('--way', '-way', default='valid', type=str, help='train or valid')
parser.add_argument('--experiment_name', '-name', default='prognostic_res_50', help='experiment name')

parser.add_argument('--type_key', '-type_key', default='tumor', type=str, help='tumor or tumor_beside or fibrous_tissue')
args = parser.parse_args()
def abnormal_KNN(train_npy, test_npy):
    clf_name = 'kNN'
    clf = KNN()
    train_npy = np.array(train_npy).reshape(-1, 1)
    clf.fit(train_npy)

    test_npy = np.array(test_npy).reshape(-1, 1)
    y_test_pred = clf.predict(test_npy)
    y_test_scores = clf.decision_function(test_npy)
    return y_test_pred





# loading checkpoint
alpha = 0.04
factor = args.factor
way = args.way

file_pth = './PrognosticVis/' + args.experiment_name
if not os.path.exists(file_pth):
    os.mkdir(file_pth)
d_pth = args.data_path
npy_pth = file_pth + '/hdf5/' + factor + '/' + way
npy_pth_train = file_pth + '/hdf5/' + factor + '/train'
img_pth = file_pth + '/img_no_abnormal/' + factor + '/' + way
json_pth = file_pth + '/path_json/'
csv_path = file_pth + '/csv/'
if not os.path.exists(file_pth + '/hdf5'):
    os.mkdir(file_pth + '/hdf5')
if not os.path.exists(file_pth + '/hdf5/' + factor):
    os.mkdir(file_pth + '/hdf5/' + factor)
if not os.path.exists(npy_pth):
    os.mkdir(npy_pth)
if not os.path.exists(file_pth + '/img_no_abnormal'):
    os.mkdir(file_pth + '/img_no_abnormal')
if not os.path.exists(file_pth + '/img_no_abnormal/' + factor):
    os.mkdir(file_pth + '/img_no_abnormal/' + factor)
if not os.path.exists(img_pth):
    os.mkdir(img_pth)
if not os.path.exists(json_pth):
    os.mkdir(json_pth)
if not os.path.exists(csv_path):
    os.mkdir(csv_path)
json_name = factor + '_' + way + '.json'
with open(json_pth + json_name, 'r') as f:
    dict = json.load(f)
json_name_train = factor + '_train.json'
with open(json_pth + json_name_train, 'r') as f:
    dict_train = json.load(f)
csv_name = 'csv_' + factor + '_' + way + '.csv'
df = pd.read_csv(csv_path + csv_name)

prediction_mean = df.Prediction.mean()
index = 0

abnormal = []
name_list = []
train_min_npy = []
train_max_npy = []
for img_name in dict_train['img_pth'].keys():
    npy = np.load(npy_pth_train + '/' + img_name + '.npy')
    train_min_npy.append(np.percentile(npy, 5))
    train_max_npy.append(np.percentile(npy, 95))
test_min_npy = []
test_max_npy = []
for img_name in dict['img_pth'].keys():
    name_list.append(img_name)
    npy = np.load(npy_pth + '/' + img_name + '.npy')
    test_min_npy.append(np.percentile(npy, 5))
    test_max_npy.append(np.percentile(npy, 95))
max_test_pred = abnormal_KNN(train_max_npy, test_max_npy)
min_test_pred = abnormal_KNN(train_min_npy, test_min_npy)
max_npy = np.array(test_max_npy)
min_npy = np.array(test_min_npy)

print(min_npy.shape)
npy_min = np.percentile(np.array(min_npy), 5)
npy_max = np.percentile(np.array(max_npy), 95)
print(npy_min, npy_max)
for img_name in dict['img_pth'].keys():
    if img_name in abnormal:
        index += 1
        continue
    img_path = os.path.join(img_pth, img_name)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    pth = dict['img_pth'][img_name]
    imgs = []
    for p in pth:

        im = cv2.imread(str(p[0]))

        imgs.append(im)
    npy = np.load(npy_pth + '/' + img_name + '.npy')
    npy = (npy - npy_min) / (npy_max - npy_min)
    npy[npy < 0] = 0
    npy[npy > 1] = 1
    predict = df[df['name'] == img_name].Prediction
    index += 1

    for i in range(len(npy)):

        img = cv2.resize(npy[i], (256, 256))
        img = np.array(img) * 255
        img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_JET)
        img_weighted = cv2.addWeighted(imgs[i], 0.5, img, 0.6, 0)
        img_new = np.hstack((img_weighted, imgs[i]))
        save_name = pth[i][0].split('/')[-2] + '_' + pth[i][0].split('/')[-1]
        cv2.imwrite(img_path + '/' + save_name, img_new)
print("abnormal image:", abnormal)
scatter = plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.scatter([i for i in range(len(min_npy))], min_npy)
plt.xlabel('point num')
plt.ylabel('min-value')
plt.title('min-value scatter')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.subplot(222)
plt.scatter([i for i in range(len(max_npy))], max_npy)
plt.xlabel('point num')
plt.ylabel('max-value')
plt.title('max-value scatter')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
print("save scatter picture")
plt.savefig(file_pth + '/scatter-' + factor + '_' + way + '.jpg')
