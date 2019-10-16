import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd
import copy
import time
import sys
import scipy as sc
import scipy.linalg as scl

def grad_AD_double(a, b, M, reg, niter, tresh):
    """Gradient with automatic differentiation."""
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    u = torch.ones(n, dtype=torch.float64) / n
    v = torch.ones(m, dtype=torch.float64) / m

    K = torch.exp(-M / reg)

    x = torch.tensor(a, dtype=torch.double, requires_grad=True)

    Kp = (1/x).view(n, 1) * K
    cpt = 0
    err = 1
    
    t1 = time.time()
    while err > tresh and cpt < niter:
        uprev = u
        vprev = v

        KtransposeU = torch.mm(torch.transpose(K, 0, 1), u.view(n, 1))
        v = torch.div(b, KtransposeU.view(m, 1))
        u = 1. / torch.mm(Kp, v.view(m, 1))

        if (KtransposeU == 0).max() or torch.isnan(u).max() \
            or torch.isnan(v).max() or (u == float('inf')).max():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            err = torch.sum((u - uprev) ** 2) / torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2) / torch.sum(v ** 2)

        cpt = cpt + 1

    T = u.view(n, 1) * K * v.view(1, m)
    cost = torch.sum(T * M)
    #cost.requires_grad_()
    cost.backward()
    grad = x.grad
    normalizer = torch.dot(torch.ones(n, dtype=torch.float64), grad.squeeze()) * torch.ones([n,1], dtype=torch.float64) / n
    #normalizer = normalizer.reshape([normalizer.size(), 1])
    grad_norm = grad - normalizer
    return T, cost, grad_norm

def gradient_sinkhorn_dual(a, b, M, reg, niter, tresh):
    """Gradient by simply taking the dual optimum.
    This is explained in:

    """
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    u = torch.ones(n, dtype=torch.float64) / n
    v = torch.ones(m, dtype=torch.float64) / m

    K = torch.exp(-M / reg)

    x = torch.tensor(a, dtype=torch.double, requires_grad=True)

    Kp = (1/x).view(n, 1) * K
    cpt = 0
    err = 1
    
    t1 = time.time()
    while err > tresh and cpt < niter:
        uprev = u
        vprev = v

        KtransposeU = torch.mm(torch.transpose(K, 0, 1), u.view(n, 1))
        v = torch.div(b, KtransposeU.view(m, 1))
        u = 1. / torch.mm(Kp, v.view(m, 1))

        if (KtransposeU == 0).max() or torch.isnan(u).max() \
            or torch.isnan(v).max() or (u == float('inf')).max():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            err = torch.sum((u - uprev) ** 2) / torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2) / torch.sum(v ** 2)

        cpt = cpt + 1

    T = u.view(n, 1) * K * v.view(1, m)
    cost = torch.sum(T * M)
    #on = torch.ones([n,1], dtype=torch.float64)
    grad = -reg*torch.log(u)
    #normalizer = reg*(torch.sum(torch.log(u))/n)
    #normalizer = normalizer.reshape([normalizer.size(), 1])
    one = torch.ones(n,dtype=torch.float64)
    grad = -(grad.reshape([n, 1])  - one * torch.dot(grad.squeeze(), one) / n)
    grad_norm = grad.reshape([n, 1]) 
    return T, cost, grad_norm


def gradient_chol(a, b, M, reg, numIter, tresh):
    """Compute gradient with closed formula using cholesky factorization."""
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    #T = ot.sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=numIter,
                #    stopThr=tresh)
    t1 = time.time()
    T, cost, u, v = Tpytorch(a, b, M, reg, numIter, tresh)
    M = M.data.numpy()
    t2 = time.time()
    print('Sinkhorn time: ', str(a.size()[0]), 'x', str(b.size()[0]), t2 - t1, 'sec')

    # if m < n we use Sherman Woodbury formula
    if m < n:
        D1i = 1/(np.sum(T, axis=1))
        D2 = np.sum(T[:, 0:m-1], axis=0)
        L = T*M
        f = -np.sum(L, axis=1) + T[:, 0:m-1] @ ((np.sum(L[:, 0:m-1].T, axis=1)) / D2)
        grada = D1i * f
        TDhalf = np.multiply(T[:, 0:m - 1].T, np.sqrt(D1i))
        K = np.diag(D2) - TDhalf @ TDhalf.T

        Lchol = scl.cho_factor(K+1e-15*np.eye(K.shape[0]), lower=True)

        grada = grada + D1i * (T[:, 0:m-1] @ scl.cho_solve(Lchol, T[:, 0:m-1].T @ grada))

    else:
        D1 = np.sum(T, axis=1)
        D2i = 1 / (np.sum(T[:,0:m-1], axis=0))
        #D2i[D2i<0] = sys.float_info.epsilon

        L = T * M
        f = -np.sum(L, axis=1) + T[:, 0:m - 1] @ ((np.sum(L[:, 0:m - 1].T, axis=1)) * D2i)
        TDhalf = np.multiply(T[:, 0:m - 1], np.sqrt(D2i))
        K = np.diag(D1) - TDhalf @ TDhalf.T

        #print(str(np.isnan(K.sum())))
        #print(str(np.isinf(K.sum())))
        Lchol = scl.cho_factor(K + 1e-15 * np.eye(K.shape[0]), lower=True)

        grada = scl.cho_solve(Lchol, f)

    grada = -(grada.reshape([n, 1]) - np.ones([n,1]) * np.dot(grada.squeeze(), np.ones([n,1])) / n)
    return T, cost, grada


def gradient_chol_pytorch(a, b, M, reg, numIter, tresh):
    """Compute gradient with closed formula using cholesky factorization."""
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    #T = ot.sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=numIter,
                #    stopThr=tresh)
    T, cost, u, v = TpytorchGPU(a, b, M, reg, numIter, tresh)
    M = M.data.numpy()

    print("Contains negative? " + str(np.sum(T<0)))

    print(str(np.isnan(T.sum())))
    print(str(np.isnan(M.sum())))
    print(str(np.isinf(T.sum())))
    print(str(np.isinf(M.sum())))
    
    # if m < n we use Sherman Woodbury formula
    if m < n:
        D1i = 1/(np.sum(T, axis=1))
        D2 = np.sum(T[:, 0:m-1], axis=0)
        L = T*M
        f = -np.sum(L, axis=1) + T[:, 0:m-1] @ ((np.sum(L[:, 0:m-1].T, axis=1)) / D2)
        grada = D1i * f
        TDhalf = np.multiply(T[:, 0:m - 1].T, np.sqrt(D1i))
        K = np.diag(D2) - TDhalf @ TDhalf.T

        Lchol = scl.cho_factor(K+1e-15*np.eye(K.shape[0]), lower=True)

        grada = grada + D1i * (T[:, 0:m-1] @ scl.cho_solve(Lchol, T[:, 0:m-1].T @ grada))

    else:
        D1 = np.sum(T, axis=1)
        D2i = 1 / (np.sum(T[:,0:m-1], axis=0))
        #D2i[D2i<0] = sys.float_info.epsilon

        L = T * M
        f = -np.sum(L, axis=1) + T[:, 0:m - 1] @ ((np.sum(L[:, 0:m - 1].T, axis=1)) * D2i)
        TDhalf = np.multiply(T[:, 0:m - 1], np.sqrt(D2i))
        K = np.diag(D1) - TDhalf @ TDhalf.T

        #print(str(np.isnan(K.sum())))
        #print(str(np.isinf(K.sum())))
        Lchol = scl.cho_factor(K + 1e-15 * np.eye(K.shape[0]), lower=True)

        grada = scl.cho_solve(Lchol, f)

    grada = -(grada - np.ones(n) * np.dot(grada.squeeze(), np.ones(n)) / n)
    return T, cost, grada

# we implement the sharp sinkhorn
def Tpytorch(a, b, M, reg, niter, tresh):
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    u = torch.ones(n, dtype=torch.float64)/n
    v = torch.ones(m, dtype=torch.float64)/m

    tM = torch.DoubleTensor(M)
    K = torch.exp(-tM/reg)

    x = torch.tensor(a, dtype=torch.double, requires_grad=False)
    y = torch.DoubleTensor(b)
        
    Kp = (1/x).view(n,1) * K
    
    cpt = 0
    err = 1

    while (err > tresh and cpt < niter):
        uprev = u
        vprev = v
        KtransposeU = torch.mm(torch.transpose(K,0,1),u.view(n,1))
        v = torch.div(y, KtransposeU.view(m, 1))
        u = 1./torch.mm(Kp, v.view(m,1))

        if (KtransposeU == 0).any() or torch.isnan(u).any()\
             or torch.isnan(v).any() or (u == float('inf')).any():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.sum( (u - uprev) ** 2)/ torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2)/torch.sum(v ** 2)

        cpt = cpt + 1

    T = u.view(n, 1) * K * v.view(1, m)
    T = T.data.numpy()
    TT = copy.deepcopy(T)
    cost = np.sum(T * M.data.numpy())
    return TT, cost, u, v

# when a,b,M is in GPU
def TpytorchGPU(a, b, M, reg, niter, tresh):
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    u = torch.ones(n, dtype=torch.float64).cuda()/n
    v = torch.ones(m, dtype=torch.float64).cuda()/m

    tM = M
    K = torch.exp(-tM/reg)

    x = a
    y = b
    
    Kp = (1/x).view(n,1) * K
    cpt = 0
    err = 1

    while (err > tresh and cpt < niter):
        uprev = u
        vprev = v
        KtransposeU = torch.mm(torch.transpose(K,0,1),u.view(n,1))
        v = y / KtransposeU.view(m)
        u = 1./torch.mm(Kp, v.view(m,1))

        if (KtransposeU == 0).any() or torch.isnan(u).any()\
             or torch.isnan(v).any() or (u == float('inf')).any():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.sum( (u - uprev) ** 2)/ torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2)/torch.sum(v ** 2)

        cpt = cpt + 1

    T = u.view(n, 1) * K * v.view(1, m)
    #T = T.data.cpu().numpy()
    #TT = np.copy.deepcopy(T)
    return T


def grad_AD_double_GPU(a, b, M, reg, niter, tresh):
    """Gradient with automatic differentiation."""
    n = a.shape[0]
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]

    u = torch.ones(n, dtype=torch.float64).cuda() / n
    v = torch.ones(m, dtype=torch.float64).cuda() / m

    K = torch.exp(-M / reg)
    
    #x = torch.tensor(a, dtype=torch.double, requires_grad=True, device="cuda").clone().detach()
    x = a
    x = x.requires_grad_()
    
    Kp = (1/x).view(n, 1) * K
    
    cpt = 0
    err = 1
    t1 = time.time()
    while err > tresh and cpt < niter:
        uprev = u
        vprev = v

        KtransposeU = torch.mm(torch.transpose(K,0,1),u.view(n,1))
        v = b / KtransposeU.view(m)
        u = 1./torch.mm(Kp, v.view(m,1))

        if (KtransposeU == 0).max() or torch.isnan(u).max() \
            or torch.isnan(v).max() or (u == float('inf')).max():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            err = torch.sum((u - uprev) ** 2) / torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2) / torch.sum(v ** 2)

        cpt = cpt + 1

    T = u.view(n, 1) * K * v.view(1, m)
    cost = torch.sum(T * M)
    cost.backward()
    grad = a.grad
    grad_norm = (grad - torch.dot(torch.ones(n, dtype=torch.float64).cuda(), grad.squeeze()) * torch.ones(n, dtype=torch.float64).cuda() / n)
    return T, cost, grad_norm

def grad_AD_x_double_GPU(a, b, x, y, reg, niter, tresh):
    """Gradient with automatic differentiation."""
    n = a.shape[0]
    m = b.shape[0]

    xv = Variable(x, requires_grad=True)

    M = dmat_cpu(xv,y)
    #M = ot.utils.dist0(n)
    #M = M1/M1.max()

    u = torch.ones(n, dtype=torch.float64).cpu() / n
    v = torch.ones(m, dtype=torch.float64).cpu() / m

    K = torch.exp(-M / reg)
        
    Kp = (1/a).view(n, 1) * K
    
    cpt = 0
    err = 1
    t1 = time.time()
    while err > tresh and cpt < niter:
        uprev = u
        vprev = v

        KtransposeU = torch.mm(torch.transpose(K, 0, 1), u.view(n, 1))
        v = torch.div(b, KtransposeU.view(m, 1))
        u = 1. / torch.mm(Kp, v.view(m, 1))
        
        if (KtransposeU == 0).max() or torch.isnan(u).max() \
            or torch.isnan(v).max() or (u == float('inf')).max():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            err = torch.sum((u - uprev) ** 2) / torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2) / torch.sum(v ** 2)

        cpt = cpt + 1

    Vv = v.view(1, m)
    T = u.view(n, 1) * K * v.view(1, m)
    cost = torch.sum(T * M)
    cost.backward()

    grad = xv.grad
    #grad = -(xv.grad.squeeze()-np.ones(m)*np.dot(xv.grad, np.ones(m))/m)
    #xv = xv.grad
    return T, cost, grad

def grad_AD_x_double_anal_GPU(a, b, x, y, reg, niter, tresh):
    """Gradient with automatic differentiation."""
    n = a.shape[0]
    m = b.shape[0]

    xv = x

    M = dmat_cpu(xv,y)
    #M = ot.utils.dist0(n)
    #M = M1/M1.max()

    u = torch.ones(n, dtype=torch.float64).cpu() / n
    v = torch.ones(m, dtype=torch.float64).cpu() / m

    K = torch.exp(-M / reg)
        
    Kp = (1/a).view(n, 1) * K
    
    cpt = 0
    err = 1
    t1 = time.time()
    while err > tresh and cpt < niter:
        uprev = u
        vprev = v

        KtransposeU = torch.mm(torch.transpose(K, 0, 1), u.view(n, 1))
        v = torch.div(b, KtransposeU.view(m, 1))
        u = 1. / torch.mm(Kp, v.view(m, 1))
        
        if (KtransposeU == 0).max() or torch.isnan(u).max() \
            or torch.isnan(v).max() or (u == float('inf')).max():

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        if cpt % 10 == 0:
            err = torch.sum((u - uprev) ** 2) / torch.sum(u ** 2) + \
                torch.sum((v - vprev) ** 2) / torch.sum(v ** 2)

        cpt = cpt + 1

    Vv = v.view(1, m)
    T = u.view(n, 1) * K * v.view(1, m)
    cost = torch.sum(T * M)
    
    #torch.mm(torch.mm(U,(1-(1/reg)*M)*K),V)
    #grad = zeros([n,1])
    #for i in range(n):
    #    for j in range(n):
    #        grad = 2*T[i,j]*(x[i]-y[j])

    return T, cost, grad


# these are numpy
def gradient_b(a, b, M, reg, numIter, tresh):
    n = a.shape[0]
    lam = 1/reg
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]
    #t1 = time.time()
    T = ot.sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=numIter, stopThr=tresh)
    #T = Tpytorch(w, a, M, reg, numIter, tresh)
    #t2 = time.time()
    #print('time for Sink ', t2-t1)

    #t1 = time.time()

    D2 = 1/(np.sum(T[0:n-1,:], axis=1)) # 1/bar a
    D1i = np.sum(T, axis=0) # b
    L = T*M
    K = (np.diag(D1i) - T[0:n-1,:].T @ np.diag(D2) @ T[0:n-1,:])
    gradb = np.linalg.solve(K, -np.sum(L.T,axis = 1) + T[0:n-1,:].T @ (D2* np.sum(L[0:n-1,:],axis = 1)) )
    #t2 = time.time()
    gradb = -(gradb-np.ones(m)*np.dot(gradb, np.ones(m))/m)
    #t2 = time.time()
    #print('time altre operaz ', t2 - t1)
    return gradb


def gradient_chol_b(a, b, M, reg, numIter, tresh):
    n = a.shape[0]
    lam = 1/reg
    m = b.shape[0]
    assert n == M.shape[0] and m == M.shape[1]
    #t1 = time.time()
    T = ot.sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=numIter, stopThr=tresh)
    #T = Tpytorch(w, a, M, reg, numIter, tresh)
    #t2 = time.time()
    #print('time for Sink ', t2-t1)

    #t1 = time.time()

    D2 = 1/(np.sum(T[0:n-1,:], axis=1)) # 1/bar a
    D1i = np.sum(T, axis=0) # b
    L = T*M
    f = -np.sum(L[:,:], axis=0) + T[0:n-1,:].T @ ((np.sum(L[0:n-1,:], axis=1))*D2)
    gradb = (1/D1i) * f
    TDhalf = np.multiply(T[0:n-1,:], np.sqrt(1/D1i))
    K = np.diag(1/D2) - TDhalf @ TDhalf.T
    try:
        Lchol = sla.cho_factor(K, lower=True)
    except:
        Lchol = sla.cho_factor(K+1e-15*np.eye(K.shape[0]), lower=True)

    gradb = gradb + (1/D1i) * (T[0:n-1,:].T @ sla.cho_solve(Lchol, T[0:n-1,:] @ gradb))
    gradb = -(gradb-np.ones(m)*np.dot(gradb, np.ones(m))/m)
    #t2 = time.time()
    #print('time altre operaz ', t2 - t1)
    return gradb