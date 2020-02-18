import torch
import torch.nn as nn
import numpy as np


def loss_fn(predict, target):
    target = one_hot_embedding(target.cpu()).cuda()

    loss = nn.BCEWithLogitsLoss()(predict, target)
    return loss, target


def one_hot_embedding(labels, num_classes=4):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [B,N].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    B, N = labels.size()
    # labels = labels.view(-1, 1)  # [B,N]->[B*N,1]
    labels = labels.view(int(B * N), 1)
    y = torch.FloatTensor(labels.size()[0], num_classes)  # [B*N, D]
    y.zero_()
    y.scatter_(1, labels, 1)
    return y  # [B*N, D]
