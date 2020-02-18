import torch
from torch.optim import SGD


class _BaseLaycaSGD(SGD):

    @staticmethod
    def layca_update(p, d_p, lr):
        raise NotImplementedError()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                self.layca_update(p, d_p, lr=group['lr'])

        return loss


class MinimalLaycaSGD(_BaseLaycaSGD):
    """
    Only step 2 and 3 from Algorithm 2 (page 17)
    """

    @staticmethod
    def layca_update(p, d_p, lr):
        # 2 - rotation-based normalization
        # d_p <- d_p * norm_2(p) / (norm_2(d_p) + eps)
        d_p.mul_(torch.norm(p) / (torch.norm(d_p) + 1e-10))
        # 3 - update
        p.data.add_(-lr, d_p)


class LaycaSGD(_BaseLaycaSGD):
    """
    All operations from Algorithm 2 (page 17)
    """

    @staticmethod
    def layca_update(p, d_p, lr):
        # 1 - project step on space orthogonal to p
        # d_p <- d_p - (d_p . p) * p / norm(p)
        norm_p = torch.norm(p)
        dot = torch.sum(d_p * p)
        d_p.sub_(dot * p / (norm_p ** 2 + 1e-10))
        # 2 - rotation-based normalization
        # d_p <- d_p * norm_2(p) / (norm_2(d_p) + eps)
        d_p.mul_(norm_p / (torch.norm(d_p) + 1e-10))
        # 3 - update
        p.data.add_(-lr, d_p)
        # 4 - project weights back on sphere
        p.data.mul_(norm_p / (torch.norm(p) + 1e-10))