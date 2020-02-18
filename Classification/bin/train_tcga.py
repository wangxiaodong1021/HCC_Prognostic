import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
# from torch import optim
from torch.nn import DataParallel, functional as F
from torch.utils.data import DataLoader
import numpy as np

torch.set_printoptions(threshold=np.inf)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from Classification.data.image_producer_tcga import GridImageDataset

from Classification.model import MODELS  # noqa
import optim


cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', default='./configs/resnet18_tcga.json', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--ckpt_path_save', '-ckpt_s', default='./model/', help='checkpoint path to save')
parser.add_argument('--log_path', '-lp', default='./log/', help='log path')
parser.add_argument('--num_workers', default=6, type=int, help='number of workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='2,3,4,5', type=str, help='comma separated indices of GPU to use,'
                                                                      ' e.g. 0,1 for using GPU_0'
                                                                      ' and GPU_1, default 0.')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=200, type=int, help='end epoch')
parser.add_argument('--ckpt', '-ckpt', default='./model/ibn_18_0/8.ckpt', help='checkpoint path')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--experiment_id', '-eid', default='0', help='experiment id')
parser.add_argument('--experiment_name', '-name', default='MobileNetV2', help='experiment name')

use_cuda = True
args = parser.parse_args()
device = torch.device("cuda" if use_cuda else "cpu")

log_path = os.path.join(args.log_path, args.experiment_name + "_" + str(args.experiment_id))
print("log_path:", log_path)

ckpt_path_save = os.path.join(args.ckpt_path_save, args.experiment_name + "_" + str(args.experiment_id))
if not os.path.exists(ckpt_path_save):
    os.mkdir(ckpt_path_save)
print("ckpt_path_save:", ckpt_path_save)


def load_checkpoint(args, net):
    print("Use ckpt: ", args.ckpt)
    assert len(args.ckpt) != 0, "Please input a valid ckpt_path"
    net_dict = net.state_dict()
    checkpoint = torch.load(args.ckpt)
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and ("fc" not in k) and ('crf' not in k)}

    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    return net


def adjust_learning_rate(optimizer, epoch, cfg):
    """decrease the learning rate at 200 and 300 epoch"""
    lr = cfg['lr']
    if epoch >= 20:
        lr /= 10
    if epoch >= 60:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def train_epoch(epoch, summary, cfg, model, loss_fn, optimizer, dataloader_train):
    model.train()
    steps = len(dataloader_train)
    batch_size = dataloader_train.batch_size
    grid_size = dataloader_train.dataset._grid_size
    dataiter = iter(dataloader_train)
    time_now = time.time()
    loss_sum = 0
    acc_sum = 0
    lr = adjust_learning_rate(optimizer, epoch, cfg)

    TP = 0
    FP = 0
    FN = 0
    summary['epoch'] = epoch
    print("steps:", steps)
    for step in range(steps):

        data, target, _ = next(dataiter)
        data = data.to(device)
        target = target.to(device)

        output,_ = model(data)
        target = target.view(-1)
        output = output.view(-1)

        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.sigmoid()
        predicts = torch.zeros_like(target)
        predicts[output > 0.5] = 1

        TP += (predicts[target == 1] == 1).type(torch.cuda.FloatTensor).sum().data.item()
        FP += (predicts[target == 0] == 1).type(torch.cuda.FloatTensor).sum().data.item()
        FN += (predicts[target == 1] == 0).type(torch.cuda.FloatTensor).sum().data.item()
        acc_data = (predicts == target).type(
            torch.cuda.FloatTensor).sum().data.item() * 1.0 / (
                           batch_size * grid_size)
        acc = (predicts == target).type(torch.cuda.FloatTensor).sum().item()
        loss_data = loss.item()
        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                                                    summary['step'] + 1, loss_data, acc_data, time_spent))


        summary['step'] += 1

        loss_sum += loss_data
        acc_sum += acc
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    summary['Precision'] = Precision
    summary['Recall'] = Recall
    summary['F1'] = F1
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / (steps * (batch_size * grid_size))
    summary['epoch'] += 1
    summary['lr'] = lr

    return summary


def valid_epoch(summary, summary_writer, epoch, model, loss_fn, dataloader_valid):
    model.eval()
    dataloader = [dataloader_valid]

    time_now = time.time()
    loss_sum = 0
    acc_sum = 0
    count = 0
    steps = 0
    steps_count = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(dataloader)):
        steps = len(dataloader[i])
        batch_size = dataloader[i].batch_size
        grid_size = dataloader[i].dataset._grid_size
        dataiter = iter(dataloader[i])
        with torch.no_grad():
            acc_tmp = 0
            loss_tmp = 0
            for step in range(steps):
                data, target, _ = next(dataiter)
                data = data.to(device)
                target = target.to(device)

                output,_ = model(data)

                target = target.view(-1)
                output = output.view(-1)

                loss = loss_fn(output, target)
                output = output.sigmoid()
                predicts = torch.zeros_like(target)
                predicts[output > 0.5] = 1

                TP += (predicts[target == 1] == 1).type(torch.cuda.FloatTensor).sum().data.item()
                FP += (predicts[target == 0] == 1).type(torch.cuda.FloatTensor).sum().data.item()
                FN += (predicts[target == 1] == 0).type(torch.cuda.FloatTensor).sum().data.item()
                acc_data = (predicts == target).type(
                    torch.cuda.FloatTensor).sum().item() * 1.0 / (
                                   batch_size * grid_size)
                acc = (predicts == target).type(torch.cuda.FloatTensor).sum().item()
                loss_data = loss.item()
                loss_sum += loss_data
                acc_sum += acc
                acc_tmp += acc
                loss_tmp += loss
                time_spent = time.time() - time_now
                time_now = time.time()
                logging.info(
                    '{}, data_num : {}, Step : {}, Testing Loss : {:.5f}, '
                    'Testing Acc : {:.3f}, Run Time : {:.2f}'
                        .format(
                        time.strftime("%Y-%m-%d %H:%M:%S"), str(i),
                        summary['step'] + 1, loss_data, acc_data, time_spent))
                summary['step'] += 1

            count += steps * (batch_size * grid_size)
            steps_count += steps
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    summary['Precision'] = Precision
    summary['Recall'] = Recall
    summary['F1'] = F1
    summary['loss'] = loss_sum / steps_count
    summary['acc'] = acc_sum / count

    return summary



def run():
    with open(args.cfg_path) as f:
        cfg = json.load(f)



    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cfg['train_batch_size'] * num_GPU
    batch_size_valid = cfg['test_batch_size'] * num_GPU
    num_workers = args.num_workers * num_GPU

    data_path = cfg['data_path_40']


    if cfg['image_size'] % cfg['patch_size'] != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side
    model = MODELS[cfg['model']](num_classes=1, num_nodes=grid_size, use_crf=cfg['use_crf'])
    if args.resume:
        model = load_checkpoint(args, model)
    model = DataParallel(model, device_ids=None)
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=1e-4, l2_reg=False)



    summary_train = {'epoch': 0, 'step': 0, 'fp': 0, 'tp': 0, 'Neg': 0, 'Pos': 0}
    summary_valid = {'loss': float('inf'), 'step': 0, 'acc': 0}
    summary_writer = SummaryWriter(log_path)
    loss_valid_best = float('inf')

    tumor_all = []
    paracancerous_all = []
    for epoch in range(args.start_epoch, args.end_epoch):


        dataset_train = GridImageDataset(data_path,
                                             cfg['json_path_train'],
                                             cfg['image_size'],
                                             cfg['patch_size'],
                                             cfg['crop_size'],
                                             rand_list=[])
        dataloader_train = DataLoader(dataset_train,
                                          batch_size=batch_size_train,
                                          num_workers=num_workers,
                                          drop_last=True,
                                          shuffle=True)

        dataset_valid = GridImageDataset(data_path,
                                               cfg['json_path_valid'],
                                               cfg['image_size'],
                                               cfg['patch_size'],
                                               cfg['crop_size'],
                                               way="valid")


        dataloader_valid = DataLoader(dataset_valid,
                                            batch_size=batch_size_valid,
                                            num_workers=num_workers,
                                            drop_last=True,
                                            shuffle=True)


        summary_train = train_epoch(epoch, summary_train, cfg, model,
                                    loss_fn, optimizer, dataloader_train)
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   (ckpt_path_save + '/' + str(epoch) + '.ckpt'))
        summary_writer.add_scalar(
            'train/loss', summary_train['loss'], epoch)
        summary_writer.add_scalar(
            'train/acc', summary_train['acc'], epoch)
        summary_writer.add_scalar(
            'learning_rate', summary_train['lr'], epoch
        )
        summary_writer.add_scalar(
            'train/Precision', summary_train['Precision'], epoch
        )
        summary_writer.add_scalar(
            'train/Recall', summary_train['Recall'], epoch
        )
        summary_writer.add_scalar(
            'train/F1', summary_train['F1'], epoch
        )

        if epoch % 2 == 0:

            summary_valid = valid_epoch(summary_valid, summary_writer, epoch, model, loss_fn,
                                        dataloader_valid)
            summary_writer.add_scalar(
                'valid/loss', summary_valid['loss'], epoch)
            summary_writer.add_scalar(
                'valid/acc', summary_valid['acc'], epoch)
            summary_writer.add_scalar(
                'valid/Precision', summary_valid['Precision'], epoch
            )
            summary_writer.add_scalar(
                'valid/Recall', summary_valid['Recall'], epoch
            )
            summary_writer.add_scalar(
                'valid/F1', summary_valid['F1'], epoch
            )

        # summary_writer.add_scalar('learning_rate', lr, epoch)
        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.module.state_dict()},
                       os.path.join(ckpt_path_save, 'best.ckpt'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    main()
