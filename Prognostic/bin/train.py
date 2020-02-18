import argparse
import os
import sys

import torch
import torch.utils.data
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from utils.Survival_Aanlysis import SurvivalAnalysis
from utils.RiskLayer import cox_cost
from Prognostic.data.image_producer import ImageDataset
from Prognostic.model import MODELS
from lifelines.utils import concordance_index

from utils.LaycaOptimizer import MinimalLaycaSGD, LaycaSGD
parser = argparse.ArgumentParser(description='Predicting survival time')
parser.add_argument('--data_path', '-d_p', default='./data/patch_prognostic', type=str,
                    help='data path')
parser.add_argument('--use_cuda', '-use_cuda', default='True', type=bool, help='use cuda')
parser.add_argument('--lr', '-lr', default='1e-4', type=float, help='learning rate')
parser.add_argument('--momentum', '-mom', default='0.9', type=float, help='SGD momentum')
parser.add_argument('--batch_size', '-b', default='5', type=int, help='batch size')
parser.add_argument('--num_worker', '-nw', default='2', type=int, help='num_worker')
parser.add_argument('--start', '-s', default='0', type=int, help='start epoch')
parser.add_argument('--end', '-e', default='10000', type=int, help='end epoch')
parser.add_argument('--experiment_id', '-eid', default='0', help='experiment id')
parser.add_argument('--experiment_name', '-name', default='prognostic_res_101_mixup', help='experiment name')
parser.add_argument('--ckpt_path_save', '-ckpt_s', default='./model/', help='checkpoint path to save')
parser.add_argument('--log_path', '-lp', default='./log/', help='log path to save')
parser.add_argument('--ckpt', '-ckpt', default='./', help='checkpoint path to load')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--way', '-way', default='10', type=str, help='train way, 40 10 or combinate')
parser.add_argument('--load_pth_train', '-lpth_t', default='./tensor_path', help='train tensor path to load')
parser.add_argument('--load_pth_valid', '-lpth_v', default='./tensor_path', help='valid tensor path to load')
parser.add_argument('--alpha', '-a', default='1.0', type=float, help='mixup alpha')
parser.add_argument('--device_ids', default='0,1,2,3,4', type=str, help='comma separated indices of GPU to use,'
                                                                      ' e.g. 0,1 for using GPU_0'
                                                                      ' and GPU_1, default 0.')
parser.add_argument('--drop_group', '-drop_group', default='3,4', help='drop groups')
parser.add_argument('--drop_prob', '-drop_prob', default='0.1', type=float, help='drop prob')
parser.add_argument('--freeze', '-f', action='store_true', help='Freeze convolutional layer parameters')
parser.add_argument('--type-key', '-type-key', default='tumor', type=str, help='tumor or tumor_beside or fibrous_tissue')
parser.add_argument('--experimentway', '-eway', default='prognosis', type=str, help='prognosis or replase')
parser.add_argument('--use_std', '-std', default='use', type=str, help='use std as feature, u:use, o:only, n:not use ')
parser.add_argument('--optimizer', '-o', default='a', type=str, help='choose optimizer:a(adam), s(sgd), '
                                                                     'Adadelta(Adadelta), m(MinimalLaycaSGD) '
                                                                     'or l(LaycaSGD)')
args = parser.parse_args()
cudnn.benchmark = True

log_path = os.path.join(args.log_path, args.experiment_name + "_" + str(args.experiment_id))
if not os.path.isdir(log_path):
    os.mkdir(log_path)
ckpt_path_save = os.path.join(args.ckpt_path_save, args.experiment_name + "_" + str(args.experiment_id))
if not os.path.exists(ckpt_path_save):
    os.mkdir(ckpt_path_save)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
device = torch.device("cuda" if args.use_cuda else "cpu")
num_GPU = len(args.device_ids.split(','))
batch_size_train = args.batch_size * num_GPU
batch_size_valid = args.batch_size * num_GPU
print("batch_size:",batch_size_train)
num_workers = args.num_worker * num_GPU


SA = SurvivalAnalysis()


def load_checkpoint(args, net):
    print("Use ckpt: ", args.ckpt)
    assert len(args.ckpt) != 0, "Please input a valid ckpt_path"
    checkpoint = torch.load(args.ckpt)
    pretrained_dict = checkpoint['state_dict']

    net.load_state_dict(pretrained_dict)
    return net


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """decrease the learning rate at 200 and 300 epoch"""
    lr = args.lr
    if epoch >= 20:
        lr /= 10
    if epoch >= 40:
        lr /= 10
    if epoch >= 80:
        lr /= 10
    '''warmup'''
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
        print('epoch = {}, step = {}, lr = {}'.format(epoch, step, lr))
    elif step == 0:
        print('epoch = {}, lr={}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


drop_prob = [0.] * 4
if args.drop_group:
    drop_probs = args.drop_prob
    drop_group = [int(x) for x in args.drop_group.split(',')]
    for block_group in drop_group:
        if block_group < 1 or block_group > 4:
            raise ValueError(
                'drop_group should be a comma separated list of integers'
                'between 1 and 4(drop_group:{}).'.format(args.drop_group)
            )
        drop_prob[block_group - 1] = drop_probs / 4.0 ** (4 - block_group)



if args.freeze:
    net = MODELS[('resnet50')](factor=args.way, drop_prob=drop_prob, require_grad=False).to(device)
    for param in net.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.lr, weight_decay=1e-2)
else:
    net = MODELS[('resnet50')](factor=args.way, drop_prob=drop_prob).to(device)
    if args.optimizer == 'a':
        print('use adam')
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    if args.optimizer == 's':
        print('use SGD')
        optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=5e-4)
    if args.optimizer == 'l':
        print('use LaycaSGD')
        optimizer = LaycaSGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    if args.optimizer == 'm':
        print('use MinimalLaycaSGD')
        optimizer = MinimalLaycaSGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    if args.optimizer == 'Adadelta':
        print('use Adadelta')
        optimizer = torch.optim.Adadelta(net.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=1e-4)




net = torch.nn.DataParallel(net, device_ids=None)




if args.resume:
    net = load_checkpoint(args, net)




def train(epoch, dataloader, summary):
    loss_sum = 0
    acc_sum = 0
    net.train()
    pth = ""
    length = len(dataloader)
    Prediction = torch.Tensor().to(device)
    Survival = torch.Tensor().to(device)
    Observed = torch.Tensor().to(device)

    for idx, (img, T, O, _, count) in enumerate(dataloader):
        if O.sum() == 0:
            continue
        N = O.shape[0]
        print('T:', T)
        print('O:', O)
        if args.optimizer != 'Adadelta':
            lr = adjust_learning_rate(optimizer, epoch, idx, len(dataloader))
        img = img.to(device)

        output = net(img)

        output, T, O, at_risk, failures, ties, _ = SA.calc_at_risk(output, T, O)
        print('ties:', ties)
        T = T.to(device)
        O = O.to(device)

        loss = cox_cost(output, at_risk, O.reshape((N, 1)), failures, ties)
        loss.register_hook(lambda g: print(g))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()

        Prediction = torch.cat((Prediction, output))
        Survival = torch.cat((Survival, T.float()))
        Observed = torch.cat((Observed, O.float()))

    Prediction, Survival, Observed, at_risk, failures, ties, _ = SA.calc_at_risk(Prediction, Survival.cpu(), Observed.cpu())

    CI = concordance_index(Survival.cpu().detach().numpy(), -Prediction.cpu().detach().numpy(),
                           Observed.cpu().detach().numpy())
    loss = cox_cost(Prediction, at_risk, Observed.reshape((Observed.shape[0],1)).to(device), failures, ties)
    print("loss:", loss.item(), "CI:", CI.item())
    summary['loss'] = loss.item()
    summary['CI'] = CI.item()
    summary['lr'] = optimizer.param_groups[0]['lr']
    return summary


def valid(dataloader, summary):
    net.eval()
    length = len(dataloader)
    Prediction = torch.Tensor().to(device)
    Survival = torch.Tensor().to(device)
    Observed = torch.Tensor().to(device)

    with torch.no_grad():
        for idx, (img, T, O, _, count) in enumerate(dataloader):
            N = O.shape[0]
            print('T:', T)
            print('O:', O)
            img = img.to(device)
            output = net(img)

            output, T, O, at_risk, failures, ties, _ = SA.calc_at_risk(output, T, O)

            T = T.to(device)
            O = O.to(device)

            loss = cox_cost(output, at_risk, O.reshape((N, 1)), failures, ties)
            print("loss:", loss.item())
            Prediction = torch.cat((Prediction, output))
            Survival = torch.cat((Survival, T.float()))
            Observed = torch.cat((Observed, O.float()))

    Prediction, Survival, Observed, at_risk, failures, ties, _ = SA.calc_at_risk(Prediction, Survival.cpu(), Observed.cpu())

    CI = concordance_index(Survival.cpu().detach().numpy(), -Prediction.cpu().detach().numpy(),
                           Observed.cpu().detach().numpy())
    loss = cox_cost(Prediction, at_risk, Observed.reshape((Observed.shape[0],1)).to(device), failures, ties)
    print("loss:", loss.item(), "CI:", CI.item())
    summary['loss'] = loss.item()
    summary['CI'] = CI.item()
    return summary


d_pth = args.data_path
sp = ckpt_path_save + '/' + str(args.way)
if not os.path.exists(sp):
    os.mkdir(sp)
print(d_pth)
train_data = ImageDataset(d_pth, factor=args.way, val=False, type_key=args.type_key,
                          ExperimentWay=args.experimentway)
valid_data = ImageDataset(d_pth, way="valid", factor=args.way, val=False, type_key=args.type_key,
                          ExperimentWay=args.experimentway)
print(len(train_data))
print(len(valid_data))
train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size_train,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size_valid,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               shuffle=False)
print("length:", len(train_dataloader))
summary_train = {'epoch': 0, 'fp': 0, 'tp': 0, 'Neg': 0, 'Pos': 0}
summary_valid = {'loss': float('inf'), 'acc': 0}
summary_writer = SummaryWriter(log_path)
loss_valid_best = float('inf')
for epoch in range(args.start, args.end):

    summary_train = train(epoch, train_dataloader, summary_train)

    summary_writer.add_scalar(
        'train/loss', summary_train['loss'], epoch)
    summary_writer.add_scalar(
        'train/CI', summary_train['CI'], epoch)
    if epoch % 1 == 0:
        torch.save({'epoch': summary_train['epoch'],
                    'state_dict': net.state_dict()},
                   (sp + '/' + str(epoch) + '.ckpt'))

    summary_valid = valid(valid_dataloader, summary_valid)

    summary_writer.add_scalar(
        'valid/loss', summary_valid['loss'], epoch)
    summary_writer.add_scalar(
        'valid/CI', summary_valid['CI'], epoch)
    summary_writer.add_scalar(
        'learning_rate', summary_train['lr'], epoch
    )
    print('train/loss', summary_train['loss'], epoch)
    print('train/CI', summary_train['CI'], epoch)
    print('valid/loss', float(summary_valid['loss']), epoch)
    print('valid/CI', summary_valid['CI'], epoch)

    if summary_valid['loss'] < loss_valid_best:
        loss_vd_best = summary_valid['loss']
        torch.save({'epoch': summary_train['epoch'],
                    'optimizer': optimizer.state_dict(),
                    'state_dict': net.state_dict()},
                   os.path.join(sp, 'best.ckpt'))


summary_writer.close()