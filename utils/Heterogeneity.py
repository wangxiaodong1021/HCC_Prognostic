import torch
import os
import numpy as np
import matplotlib
import itertools

matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from Prognostic.data.image_producer import ImageDataset
from Prognostic.model import MODELS

np.set_printoptions(threshold=np.inf)


def Entropy(feature_map):
    entropy = []
    for i in range(feature_map.shape[1]):
        if feature_map[:, i, :, :].max() == feature_map[:, i, :, :].min():
            print(feature_map[:, i, :, :])
        feature = feature_map[:, i, :, :].flatten()
        feature = (feature - feature.max()) / (feature.max() - feature.min())
        hist, _ = np.histogram(feature, list(np.arange(0, 1, 0.1)))
        probability = hist / len(feature)
        probability[probability == 0] = 1
        entropy.append(-probability * np.log2(probability))
    return np.array(entropy)


# loading checkpoint
alpha = 0.04
factor = '10'
way = 'valid'

ckpt_path = '/home/hdc/Liver/model/only_tumor_0/10/best.ckpt'
d_pth = '/data/pathology_2/Liver_all/patch_new'
npy_pth = '/home/hdc/Liver/Prognostic/feature_map' + '/' + str(factor)
if not os.path.isdir(npy_pth):
    os.mkdir(npy_pth)
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

valid_data = ImageDataset(d_pth, way=way, factor=factor, val=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=1,
                                               num_workers=4,
                                               drop_last=False,
                                               shuffle=False)
drop_prob = [0., 0., 0., 0.]
net = MODELS[('resnet50')](way='val', drop_prob=drop_prob).to(device)
net = torch.nn.DataParallel(net, device_ids=None)
checkpoint = torch.load(ckpt_path)
state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net = net.eval()
fc_weight = state_dict['module.fc.weight']

# hist = plt.figure(figsize=(15, 15))
# plt.hist(fc_weight.cpu(), bins=10, facecolor='blue', edgecolor='white', alpha=0.7)
# plt.xlabel('weight')
# plt.ylabel('frequency')
# plt.title('fc weight hist')
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# print("save hist picture")
# plt.savefig('weight hist-' + factor + '_' + way + '.jpg')

img_names = []
entropy_list = []
with torch.no_grad():
    for idx, (img, T, O, pth) in enumerate(valid_dataloader):
        img = img.to(device)
        T = T.to(device)
        T = T.to(device)
        O = O.to(device)
        img_name = pth[0][0].split('/')[-3]
        feature_map = net(img)
        feature_map = feature_map.squeeze(0)
        # np.save(npy_pth+'/'+img_name+'.npy',feature_map.cpu().numpy())
        entropy = Entropy(feature_map.cpu().numpy())
        entropy_list.append(entropy)
entropy_npy = np.array(entropy_list)
scatter = plt.figure(figsize=(15, 15))
a = [i for i in range(entropy_npy.shape[0])]
b = [i for i in range(entropy_npy.shape[1])]
x, y, z = [], [], []
for i, j in itertools.product(a, b):
    x.append(i)
    y.append(j)
    z.append(entropy_npy[i][j])
    print(i, j, entropy_npy[i][j])
fig = plt.figure()
ax = Axes3D(fig)
print(len(x), len(y), len(z))
ax.scatter(x, y, z)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.title('entropy scatter')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
print("save scatter picture")
plt.savefig('scatter-' + factor + '_' + way + '.jpg')
# import os
# d_pth = '/data/pathology_2/Liver_all/patch_new/x10'
# way = ["train","valid"]
# less100 = []
# for i in way:
#     pth = d_pth +'/' + i
#     for file in os.listdir(pth):
#         file_pth = pth + '/' + file + '/tumor'
#         if len(os.listdir(file_pth)) < 100:
#             less100.append(file)
# print(less100)
