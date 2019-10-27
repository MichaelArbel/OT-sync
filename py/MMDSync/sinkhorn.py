import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd
from utils import quaternion_geodesic_distance, squared_quaternion_geodesic_distance
import utils

# Adapted from ../OptimalTransportSync
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff

from geomloss import SamplesLoss




def get_loss(kernel_type,eps):
    if kernel_type=='quaternion':
        return SamplesLoss("sinkhorn", blur=eps, diameter=3.15, cost = utils.quaternion_geodesic_distance, backend= 'tensorized')
    elif kernel_type=='squared_euclidean':
        return SamplesLoss("sinkhorn", p=2, blur=eps, diameter=4., backend= 'tensorized')
    elif kernel_type=='power_quaternion':
        return SamplesLoss("sinkhorn", blur=eps, diameter=10.,cost = utils.power_quaternion_geodesic_distance, backend= 'tensorized')
    elif kernel_type=='sinkhorn_gaussian':
        return SamplesLoss("gaussian", blur=1., diameter=4., backend= 'tensorized')
    elif kernel_type=='min_squared_euclidean':
        return SamplesLoss("sinkhorn", blur=eps, diameter=10.,cost = utils.min_squared_eucliean_distance, backend= 'tensorized')
#loss = SamplesLoss("sinkhorn", blur=.05, diameter=3.15, cost = quaternion_geodesic_distance_geomloss, backend= 'tensorized')


class Sinkhorn(nn.Module):
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
    def __init__(self, kernel_type, particles,rm_map, eps=0.05):
        super(Sinkhorn, self).__init__()
        self.eps = eps
        self.kernel_type = kernel_type
        self.particles = particles
        self.rm_map = rm_map
        self.loss = get_loss(kernel_type,eps)
    def forward(self, true_data):
        # The Sinkhorn algorithm takes as input three variables :
        y = self.rm_map(self.particles.data)
        return  torch.sum(self.loss(true_data,y))


class Sinkhorn_weighted(nn.Module):
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
    def __init__(self, kernel_type, particles,rm_map, eps=0.05):
        super(Sinkhorn_weighted, self).__init__()
        self.eps = eps
        self.kernel_type = kernel_type
        self.particles = particles
        self.rm_map = rm_map
        self.loss = get_loss(kernel_type,eps)
    def forward(self, true_data,true_weights):
        # The Sinkhorn algorithm takes as input three variables :
        y, weights = self.rm_map(self.particles.data,self.particles.weights())
        return  torch.sum(self.loss(true_weights,true_data,weights,y))


# class SinkhornLoss(nn.Module):
#     r"""
#     Given two empirical measures each with :math:`P_1` locations
#     :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
#     outputs an approximation of the regularized OT cost for point clouds.
#     Args:
#         eps (float): regularization coefficient
#         max_iter (int): maximum number of Sinkhorn iterations
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
#             'mean': the sum of the output will be divided by the number of
#             elements in the output, 'sum': the output will be summed. Default: 'none'
#     Shape:
#         - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
#         - Output: :math:`(N)` or :math:`()`, depending on `reduction`
#     """
#     def __init__(self, kernel_type, particles,rm_map, eps=0.05):
#         super(SinkhornLoss, self).__init__()
#         self.eps = eps
#         self.kernel_type = kernel_type
#         self.particles = particles
#         self.rm_map = rm_map
#         self.loss = get_loss(kernel_type,eps)
#     def forward(self, true_data):
#         # The Sinkhorn algorithm takes as input three variables :
#         y = self.particles.data
#         #print(true_data)
#         return  torch.sum(self.loss(true_data,y))




class SinkhornEval(nn.Module):
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
    def __init__(self, eps, max_iter, particles_type):
        super(SinkhornEval, self).__init__()
        self.eps = eps
        self.particles_type = particles_type
        kernel_type = 'quaternion'
        self.loss = get_loss(kernel_type,eps)

    def forward(self, x, y,w_x,w_y):
        # The Sinkhorn algorithm takes as input three variables :
        if w_x is None or w_y is None:

            return  torch.mean(self.loss(x,y))
        else:
            return  torch.mean(self.loss(w_x,x,w_y,y))
    