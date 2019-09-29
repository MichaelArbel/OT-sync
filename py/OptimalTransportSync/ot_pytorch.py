import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd


def sink(M, reg, numItermax=1000, stopThr=1e-9, cuda = True):

    # we assume that no distances are null except those of the diagonal of
    # distances

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    Nini = len(a)
    Nfin = len(b)

    if cuda:
        u = Variable(torch.ones(Nini) / Nini).cuda()
        v = Variable(torch.ones(Nfin) / Nfin).cuda()
    else:
        u = Variable(torch.ones(Nini) / Nini)
        v = Variable(torch.ones(Nfin) / Nfin)

    # print(reg)

    K = torch.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        #print(T(K).size(), u.view(u.size()[0],1).size())
        KtransposeU = K.t().matmul(u)
        v = torch.div(b, KtransposeU)
        u = 1. / Kp.matmul(v)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = u.view(-1, 1) * (K * v)
            # err = (torch.sum(transp) - b).norm(1).pow(2).data[0]
            err = (torch.sum(transp,dim=0) - b).norm(2).pow(2).data.item()


        cpt += 1

    return torch.sum(u.view((-1, 1)) * K * v.view((1, -1)) * M)


def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20, cuda=True):

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if cuda:
            alpha, beta = Variable(torch.zeros(na)).cuda(), Variable(torch.zeros(nb)).cuda()
        else:
            alpha, beta = Variable(torch.zeros(na)), Variable(torch.zeros(nb))
    else:
        alpha, beta = warmstart

    if cuda:
        u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
    else:
        u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg + torch.log(u.view((na, 1))) + torch.log(v.view((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).data[0] > tau or torch.max(torch.abs(v)).data[0] > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            if cuda:
                u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
            else:
                u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp) - b).norm(1).pow(2).data[0]

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        #if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        #    # we have reached the machine precision
        #    # come back to previous solution and quit loop
        #    print('Warning: numerical errors at iteration', cpt)
        #    u = uprev
        #    v = vprev
        #    break

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v)*M)

def pairwise_distances(x, y, method='l1'):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == 'l1':
        dist = torch.abs(x - y).sum(2)
    else:
        dist = torch.pow(x - y, 2).sum(2)

    return dist.float()

def dmat(x,y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze().cuda()

    return mm   
    
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1