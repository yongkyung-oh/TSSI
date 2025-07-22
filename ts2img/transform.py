import os
import sys
import time
import datetime
import pickle
import random

import numpy as np
import pandas as pd
import cv2
import torch

from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation
from skimage.transform import resize
from scipy import signal
from scipy.signal import cwt, spectrogram
from scipy.interpolate import interp1d


def transform_TSSI(sample, size, scale, **kwargs):
    min_val, max_val = scale
    
    res_list = []
    for y in sample:
        x_len = len(y)
        x_new = np.linspace(0, x_len-1, x_len*10, endpoint=True)

        f_interpolation = interp1d(range(x_len), y, kind='linear')
        y_new = f_interpolation(x_new)
        y_new = np.array([y if y < max_val else max_val for y in y_new])
        y_new = np.array([y if y > min_val else min_val for y in y_new])

        x_edge = np.linspace(0, max(x_new), size+1, endpoint=True)
        y_edge = np.linspace(min_val, max_val, size+1, endpoint=True)
        H, xedges, yedges = np.histogram2d(x_new, y_new, bins=(x_edge, y_edge))
        H_flip = np.flipud(H.T)
        
        # normalize
        if H_flip.max() == H_flip.min():
            H_flip = H_flip.clip(0,0)
        else:
            H_flip = (H_flip-H_flip.min())/(H_flip.max()-H_flip.min()) * 255

        # binary    
        H_flip[H_flip>0] = 255
        H_flip[H_flip==0] = 0
            
        res = cv2.resize(H_flip, dsize=(size, size), interpolation=cv2.INTER_AREA)
        res_list.append(res.astype(np.uint8))
    res_tensor = torch.tensor(np.array(res_list))
    return res_tensor    


def transform_GADF(sample, size, image_size, sample_range, **kwargs):
    # image_size
    if sample.shape[1] > size:
        transformer = GramianAngularField(image_size=size, sample_range=sample_range, method='difference')
        y_new = transformer.transform(sample)
    else:
        transformer = GramianAngularField(sample_range=sample_range, method='difference')
        y_new = transformer.transform(sample)
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_GASF(sample, size, image_size, sample_range, **kwargs):
    # image_size
    if sample.shape[1] > size:
        transformer = GramianAngularField(image_size=size, sample_range=sample_range, method='summation')
        y_new = transformer.transform(sample)
    else:
        transformer = GramianAngularField(sample_range=sample_range, method='summation')
        y_new = transformer.transform(sample)
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_MTF(sample, size, image_size, n_bins, **kwargs):
    # image_size
    if sample.shape[1] > image_size:
        transformer = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
        y_new = transformer.transform(sample)
    else:
        transformer = MarkovTransitionField(n_bins=n_bins)
        y_new = transformer.transform(sample)
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_RP(sample, size, image_size, time_delay, **kwargs):
    # image_size
    if sample.shape[1] > image_size:
        transformer = RecurrencePlot(time_delay=time_delay)
        y_new = transformer.transform(sample)
        y_new = np.array([resize(y,(image_size,image_size)) for y in y_new])
    else:
        transformer = RecurrencePlot(time_delay=time_delay)
        y_new = transformer.transform(sample)
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_GS(sample, size, **kwargs):
    # 128 * 128
    def gray_scale(ts, K_len=128, s=5, P=255):
        s = round((s-1)/2)
        ts = ts[s:-s-1]   # remove the remainings

        ## make matrix
        UB = np.max(ts)
        LB = np.min(ts)
        gs_mat = np.zeros((K_len, K_len))
        for i in range(K_len):
            for j in range(K_len):
                if (UB - LB) != 0:
                    gs_mat[i,j] = round(P * (ts[i*s+j] - LB)/(UB - LB))
                else:
                    gs_mat[i,j] = round(P * (ts[i*s+j] - LB))
        return gs_mat
    
    y_new = []
    for y in sample:
        x_len = len(y)
        x_new = np.linspace(0, x_len-1, 2**9, endpoint=True)

        f_interpolation = interp1d(range(x_len), y, kind='linear')
        gs = gray_scale(f_interpolation(x_new))
        y_new.append(gs)
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_SC(sample, size, **kwargs):
    # 128 * 128
    def avg_pooling(arr, K=8, L=8):
        M, N = arr.shape
        MK = M // K
        NL = N // L
        return arr[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))
    
    y_new = []
    for y in sample:
        x_len = len(y)
        x_new = np.linspace(0, x_len-1, 2**9, endpoint=True)

        f_interpolation = interp1d(range(x_len), y, kind='linear')
        sc = cwt(f_interpolation(x_new).reshape(-1), signal.ricker, np.arange(128)+1)
        y_new.append(avg_pooling(sc, K=1, L=4))
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_SP(sample, size, **kwargs):
    # 128 * 33
    y_new = []
    for y in sample:
        x_len = len(y)
        x_new = np.linspace(0, x_len-1, 2**9, endpoint=True)

        f_interpolation = interp1d(range(x_len), y, kind='linear')
        f, t, sp = spectrogram(f_interpolation(x_new), fs=1, window='hann', nperseg=256-1, noverlap=256-9, nfft=None)
        y_new.append(sp)
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_RPM(sample, size, **kwargs):
    # 128 * 128
    def avg_pooling(arr, K=8, L=8):
        M, N = arr.shape
        MK = M // K
        NL = N // L
        return arr[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))

    transformer = PiecewiseAggregateApproximation()
    sample_paa = transformer.transform(sample)
    
    y_new = []
    for y in sample_paa:
        x_len = len(y)
        x_new = np.linspace(0, x_len-1, 2**9, endpoint=True)

        f_interpolation = interp1d(range(x_len), y, kind='linear')
        intp = f_interpolation(x_new)

        rpm = np.repeat(intp.reshape(-1,1), 2**9, axis=1) - intp
        rpm = (rpm-rpm.min())/(rpm.max()-rpm.min())
        y_new.append(avg_pooling(rpm, K=4, L=4))
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    


def transform_SRPM(sample, size, **kwargs):
    # 128 * 128
    def avg_pooling(arr, K=8, L=8):
        M, N = arr.shape
        MK = M // K
        NL = N // L
        return arr[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))
    
    y_new = []
    for y in sample:
        x_len = len(y)
        x_new = np.linspace(0, x_len-1, 2**9, endpoint=True)

        f_interpolation = interp1d(range(x_len), y, kind='linear')
        intp = f_interpolation(x_new)

        w = 2**3
        intp_std = np.array([np.std(intp[max(i-w,0):i+w])/(2*w+1) for i in range(len(intp))])
        intp_stdd = np.array([np.std(intp_std[max(i-w,0):i+w])/(2*w+1) for i in range(len(intp_std))])

        srpm = np.zeros((2**8,2**8))
        for i in range(2**8):
            srpm[i] = intp_stdd[i+np.arange(2**8)] - intp_stdd[i] - intp_stdd[np.arange(2**8)]
        srpm = (srpm-srpm.min())/(srpm.max()-srpm.min())
        y_new.append(avg_pooling(srpm, K=2, L=2))
    y_new = np.array([resize(y,(size,size)) for y in y_new])
    res_tensor = torch.tensor(y_new)
    return res_tensor    
