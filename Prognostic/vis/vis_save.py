import argparse
import torch
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import cv2
import json
import pandas as pd
from lifelines.utils import concordance_index

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from utils.Survival_Aanlysis import SurvivalAnalysis
from utils.RiskLayer import cox_cost
from Prognostic.data.image_producer import ImageDataset
from Prognostic.model import MODELS
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='Predicting survival time')
parser.add_argument('--data_path', '-d_p', default='/data/data_ext_1/Liver_all/prognosis/', type=str,
                    help='data path')
parser.add_argument('--experimentway', '-eway', default='prognosis', type=str, help='prognosis or replase')
parser.add_argument('--factor', '-factor', default='10', type=str, help='valid way, 40 10 or combinate')
parser.add_argument('--way', '-way', default='valid', type=str, help='train or valid')
parser.add_argument('--experiment_name', '-name', default='prognostic_res_50', help='experiment name')
parser.add_argument('--type_key', '-type_key', default='tumor', type=str, help='tumor or tumor_beside or fibrous_tissue')
parser.add_argument('--save_checkpoint', '-save', default='', type=str, help='save checkpoint path')
parser.add_argument('--checkpoint_40', '-ckpt_40', default='', type=str, help=' 40 checkpoint')
parser.add_argument('--checkpoint_10', '-ckpt_10', default='', type=str, help=' 10 checkpoint')
parser.add_argument('--checkpoint_4', '-ckpt_4', default='', type=str, help=' 4 checkpoint')
args = parser.parse_args()
def weight_vis(fc_weight):
    fc_weight = np.array(fc_weight.squeeze())
    print(fc_weight.shape)
    plt.subplot(211)
    plt.scatter(np.arange(fc_weight.shape[0]), fc_weight, linewidths=0.002)
    plt.subplot(212)
    plt.hist(fc_weight, 12)
    plt.show()
    gt = (fc_weight > 0.05).sum()
    lt = (fc_weight < -0.05).sum()
    print(gt, lt)

def save_checkpoint(checkpoint, alpha=0.05):
    state_dict = checkpoint['state_dict']
    fc_weight = state_dict['module.fc.weight']
    fc_weight[fc_weight < alpha] = 0
    print(alpha, torch.nonzero(fc_weight).size(0), torch.unique(fc_weight))
    state_dict["module.fc.weight"] = fc_weight
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, args.save_checkpoint + '/' + str(alpha) + '.ckpt')


def CAM_new(logits, weight):
    logits = logits.numpy()  # (num,1024,16,16)
    weight = weight.numpy()  # (1,2048)
    logits = logits.squeeze()
    # weight = weight.reshape(2048, -1)
    weight = weight.reshape(1024, -1)
    logits = np.transpose(logits, (0, 2, 3, 1))
    print(logits.shape)
    lw = logits.dot(weight)
    return lw, np.min(lw), np.max(lw)


def save_feature(logits, pth, save_pth):
    logits = logits.numpy().squeeze()
    file_name = pth[0][0].split('/')[-3]
    save_path = os.path.join(save_pth, file_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for i in range(0, len(logits), 10):
        for j in range(20):
            img_name = pth[i][0].split('/')[-2] + '_' + pth[i][0].split('/')[-1].strip('.png') + '_' + str(j) + '.png'
            print(save_path + '/' + img_name)
            img = np.array((logits[i][j] - np.min(logits[i][j])) / (np.max(logits[i][j]) - np.min(logits[i][j])))
            img = cv2.resize(img, (256, 256))
            img = np.array(img) * 255
            img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_JET)
            cv2.imwrite(save_path + '/' + img_name, img)
def mkdir(pth, name, factor=-1, way=''):
    file_pth = os.path.join(pth + '/' + name)
    if not os.path.exists(file_pth):
        os.mkdir(file_pth)
    if factor != -1:
        if not os.path.exists(file_pth + '/' + factor):
            os.mkdir(file_pth + '/' + factor)
    if way != '':
        if not os.path.exists(file_pth + '/' + factor + '/' + way):
            os.mkdir(file_pth + '/' + factor + '/' + way)


# loading checkpoint
alpha = 0.04
factor = args.factor
way = args.way
type_key = args.type_key
ExperimentName = args.experiment_name
experimentway = args.experimentway

ckpt_path = {'40': args.checkpoint_40,
             '10': args.checkpoint_10,
             '4': args.checkpoint_4}

file_pth = './PrognosticVis/' + ExperimentName
if not os.path.exists(file_pth):
    os.mkdir(file_pth)
d_pth = args.data_path
logits_path = file_pth + '/logits/'
npy_pth = file_pth + '/hdf5/' + factor + '/' + way
img_pth = file_pth + '/img/' + factor + '/' + way
json_pth = file_pth + '/path_json/'
csv_path = file_pth + '/csv/'
mkdir(file_pth, 'logits')
mkdir(file_pth, 'hdf5', factor, way)
mkdir(file_pth, 'img', factor, way)
mkdir(file_pth, 'path_json')
mkdir(file_pth, 'csv')
SA = SurvivalAnalysis()
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
valid_data = ImageDataset(d_pth, way=way, val=True, factor=factor, type_key=type_key, ExperimentWay=experimentway, tcga='y')
valid_dataloader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=1,
                                               num_workers=4,
                                               drop_last=False,
                                               shuffle=False)
print(len(valid_dataloader))
drop_prob = [0., 0., 0., 0.]
net = MODELS[('resnet50')](way='val', drop_prob=drop_prob).to(device)
net = torch.nn.DataParallel(net, device_ids=None)
checkpoint = torch.load(ckpt_path[factor])
state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net = net.eval()
fc_weight = state_dict['module.fc.weight']

dict = {}
dict['img_pth'] = {}
img_min = float('inf')
img_max = float('-inf')
Prediction = torch.Tensor().to(device)
Survival = torch.Tensor().to(device)
Observed = torch.Tensor().to(device)
img_names = []
with torch.no_grad():
    for idx, (img, T, O, pth, _) in enumerate(valid_dataloader):
        print("idx:", idx, pth[0][0].split('/')[-3], T, O)
        img = img.to(device)
        T = T.to(device)
        O = O.to(device)
        output, logits = net(img)
        feature_weight, fw_min, fw_max = CAM_new(logits.cpu(), fc_weight.cpu())
        img_name = pth[0][0].split('/')[-3]
        img_names.append(img_name)
        np.save(npy_pth + '/' + img_name + '.npy', feature_weight)
        dict['img_pth'][img_name] = pth
        img_min = min(img_min, fw_min)
        img_max = max(img_max, fw_max)
        Prediction = torch.cat((Prediction, output))
        Survival = torch.cat((Survival, T.float()))
        Observed = torch.cat((Observed, O.float()))
Prediction, Survival, Observed, at_risk, failures, ties, img_names = SA.calc_at_risk(Prediction, Survival.cpu(),
                                                                                     Observed.cpu(),
                                                                                     np.array(img_names))
CI = concordance_index(Survival.cpu().detach().numpy(), -Prediction.cpu().detach().numpy(),
                       Observed.cpu().detach().numpy())
loss = cox_cost(Prediction, at_risk, Observed.reshape((Observed.shape[0], 1)).to(device), failures, ties)
df = pd.DataFrame(
    data=np.stack(
        (Prediction.squeeze().cpu().numpy(), Survival.cpu().numpy(), Observed.cpu().numpy(), img_names), 0).T,
    columns=['Prediction', 'Survival', 'Observed', 'name'])
csv_name = 'csv_' + factor + '_' + way + '.csv'
df.to_csv(csv_path + csv_name, index=False)

dict['CI'] = CI
dict['loss'] = loss.item()
dict['min'] = str(img_min)
dict['max'] = str(img_max)
json_name = factor + '_' + way + '.json'
with open(json_pth + json_name, 'w') as f:
    json.dump(dict, f)
###plot KM
df[['Prediction']] = df[['Prediction']].astype(float)
group = pd.cut(df.Prediction, [df.Prediction.min(), df.Prediction.mean(), df.Prediction.max()],
               labels=['low-risk', 'high-risk'])
figure = plt.subplot()
kmf = KaplanMeierFitter()
df[['Survival']] = df[['Survival']].astype(float)
T_high = df[group == 'high-risk'].Survival
E_high = df[group == 'high-risk'].Observed
kmf.fit(T_high, event_observed=E_high.astype(bool), label='high-risk')
kmf.plot(ax=figure, show_censors=True)
T_low = df[group == 'low-risk'].Survival
E_low = df[group == 'low-risk'].Observed
kmf.fit(T_low, event_observed=E_low.astype(bool), label='low-risk')
kmf.plot(ax=figure, show_censors=True)
plt.legend(['Censored Data', 'Kaplan-Meier Estimate', 'Confidence interval'])
plt.title('Kaplan Meier')
plt.savefig(file_pth + '/' + 'KM-' + factor + '_' + way + '.jpg')
logrank_test_results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
logrank_test_results.print_summary()
print('p_value', logrank_test_results)
print(logrank_test_results.test_statistic)
print('CI:', CI)