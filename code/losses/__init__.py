"""
the package store the loss that user defined
"""

import torch.nn.functional as F
import torch
from math import exp
import torch.nn as nn
import numpy as np
import math
import scipy.signal as signal
from skimage.color import rgb2lab
# from losses.config import Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def ComVidVindG(ref, dist, sq):
    # initialize variables
    sigma_nsq = sq
    Num = []
    Den = []
    G = []

    # loop over 4 scales
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
############   win = GaussianWindow(N)
        win = torch.tensor([[torch.exp(torch.tensor(-(i ** 2 + j ** 2)) / torch.tensor((2 * ((0.5 * N) / 5.0) ** 2)))
                             for j in range(-N // 2, N // 2 + 1)] for i in range(-N // 2, N // 2 + 1)])
        win = win.unsqueeze(0).unsqueeze(0).to(device)  # Move filter tensor to the device
        win = win / torch.sum(win)

        if scale > 1:
            ############    ref, dist = Convolve(ref, win), Convolve(dist, win)
            ref = F.conv2d(ref, win, stride=2, padding=(N-1)//2)
            dist = F.conv2d(dist, win, stride=2, padding=(N-1)//2)
        ref = ref.to(device)  # Move input tensor to the same device
        mu1 = F.conv2d(ref, win)
        mu2 = F.conv2d(dist, win)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(ref**2, win) - mu1_sq
        sigma2_sq = F.conv2d(dist**2, win) - mu2_sq
        sigma12 = F.conv2d(ref*dist, win) - mu1_mu2

        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq<1e-10] = 0
        sv_sq[sigma1_sq<1e-10] = sigma2_sq[sigma1_sq<1e-10]
        sigma1_sq[sigma1_sq<1e-10] = 0

        g[sigma2_sq<1e-10] = 0
        sv_sq[sigma2_sq<1e-10] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=1e-10] = 1e-10

        G.append(g)
        VID = torch.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq))
        VIND = torch.log10(1 + sigma1_sq / sigma_nsq)
        Num.append(VID)
        Den.append(VIND)

    Tg1 = Num
    Tg2 = Den
    Tg3 = G

    return Tg1, Tg2, Tg3



def VIFF_Public(Im1, Im2, ImF):
    # visual noise
    sq = 0.005 * 255 * 255
    # error comparison parameter
    C = 1e-7

    b,r, s, l = Im1.shape
    # color space transformation
    if l == 3:
        T1 = rgb2lab(Im1)
        T2 = rgb2lab(Im2)
        TF = rgb2lab(ImF)
        Ix1 = T1[:, :, 0]
        Ix2 = T2[:, :, 0]
        IxF = TF[:, :, 0]
    else:
        Ix1 = Im1
        Ix2 = Im2
        IxF = ImF

    T1p = Ix1.clone().detach().to(device)
    T2p = Ix2.clone().detach().to(device)
    Trp = IxF.clone().detach().to(device)


    p = torch.tensor([1, 0, 0.15, 1]).to(device) / 2.15
    T1N, T1D, T1G = ComVidVindG(T1p, Trp, sq)
    T2N, T2D, T2G = ComVidVindG(T2p, Trp, sq)
    VID = []
    VIND = []
    # i multiscale image level
    for i in range(4):
        M_Z1 = T1N[i]
        M_Z2 = T2N[i]
        M_M1 = T1D[i]
        M_M2 = T2D[i]
        M_G1 = T1G[i]
        M_G2 = T2G[i]
        L = M_G1 < M_G2
        M_G = M_G2.clone()
        M_G[L] = M_G1[L]
        M_Z12 = M_Z2.clone()
        M_Z12[L] = M_Z1[L]
        M_M12 = M_M2.clone()
        M_M12[L] = M_M1[L]

        VID.append(torch.sum((M_Z12 + C)))
        VIND.append(torch.sum((M_M12 + C)))
    F = torch.sum(torch.tensor(VID).to(device) / torch.tensor(VIND).to(device) * p)
    # F.requires_grad = True
    return 1-F




def L1_loss(img1,img2):

    # Create the L1 loss function
    loss_fn = nn.L1Loss()
    img1=img1.to(device)
    img2 = img2.to(device)
    # Compute the loss
    loss = loss_fn(img1,img2)
    return loss
