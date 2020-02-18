import torch
from torch import nn
from torch.nn import functional as F



class CRF(nn.Module):
    def __init__(self, num_nodes, iteration=10, num_classes=1):
        """Initialize the CRF module

        Args:
            num_nodes: int, number of nodes/patches within the fully CRF
            iteration: int, number of mean field iterations, e.g. 10
        """
        super(CRF, self).__init__()
        self.num_nodes = num_nodes
        self.iteration = iteration
        self.W = nn.Parameter(torch.zeros(1, num_nodes, num_nodes))
        self.num_classes = num_classes
    def forward(self, feats, logits):
        """Performing the CRF.

        Args:
            feats: 3D tensor with the shape of
            [batch_size, num_nodes, embedding_size], where num_nodes is the
            number of patches within a grid, e.g. 9 for a 3x3 grid;
            embedding_size is the size of extracted feature representation for
            each patch from ResNet, e.g. 512
            logits: 3D tensor with shape of [batch_size, num_nodes, 1], the
            logit of each patch within the grid being tumor before CRF

        Returns:
            logits: 3D tensor with shape of [batch_size, num_nodes, 1], the
            logit of each patch within the grid being tumor after CRF
        """
        feats_norm = torch.norm(feats, p=2, dim=2, keepdim=True)
        pairwise_norm = torch.bmm(feats_norm,
                                  torch.transpose(feats_norm, 1, 2))
        pairwise_dot = torch.bmm(feats, torch.transpose(feats, 1, 2))
        # cosine similarity between feats
        pairwise_sim = pairwise_dot / pairwise_norm
        # symmetric constraint for CRF weights
        W_sym = (self.W + torch.transpose(self.W, 1, 2)) / 2
        pairwise_potential = pairwise_sim * W_sym
        unary_potential = logits.clone()
        num_classes = self.num_classes
        if num_classes > 1:
            for j in range(num_classes):
                unary_potential = logits[:, :, j].clone().unsqueeze(2)
                for i in range(self.iteration):
                    # current Q
                    probs = torch.transpose(logits[:, :, j].sigmoid().unsqueeze(2), 1, 2)
                    # taking expectation of pairwise_potential using current Q
                    pairwise_potential_E = torch.sum(
                        probs * pairwise_potential - (1 - probs) * pairwise_potential,
                        dim=2, keepdim=True)
                if j == 0:
                    logits_new = unary_potential + pairwise_potential_E
                else:
                    logits_new = torch.cat((logits_new, (unary_potential + pairwise_potential_E)), 2)
            return logits_new
        else:
            for i in range(self.iteration):
                # current Q after normalizing the logits
                probs = torch.transpose(logits.sigmoid(), 1, 2)
                # taking expectation of pairwise_potential using current Q
                pairwise_potential_E = torch.sum(
                    probs * pairwise_potential - (1 - probs) * pairwise_potential,
                    dim=2, keepdim=True)
                logits = unary_potential + pairwise_potential_E

            return logits

    def __repr__(self):
        return 'CRF(num_nodes={}, iteration={})'.format(
            self.num_nodes, self.iteration
        )
