import zipfile
import os
import pandas as pd
import math, random
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from sklearn.model_selection import StratifiedShuffleSplit
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
try:
    from scipy.fftpack import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft
from scipy.signal import lfilter
import scipy.io as sio
from scipy import signal
import gc
import h5py
from torchsummary import summary
!pip install torchmetrics
from torchmetrics import Precision, Recall, ConfusionMatrix
from torchmetrics.functional import f1_score



######################################### MRCG CODE
epsc = 0.000001

def mrcg_extract(sig, sampFreq = 32000): # Sample frequency is always 32,000 in our case
    # Code From: https://github.com/MoongMoong/MRCG_python/blob/master/MRCG_python_master/mrcg/MRCG.py
    
    beta = 1000 / np.sqrt(sum(map(lambda x:x*x,sig)) / len(sig))
    sig = sig*beta
    sig = sig.reshape(len(sig), 1)
    g = gammatone(sig, 64, sampFreq)
    cochlea1 = np.log10(cochleagram(g, int(sampFreq * 0.025), int(sampFreq * 0.010)) + 0.0000005)
    cochlea2 = np.log10(cochleagram(g, int(sampFreq * 0.200), int(sampFreq * 0.010)) + 0.0000005) # 768, x 
    cochlea1 = cochlea1[:,:]
    cochlea2 = cochlea2[:,:]
    cochlea3 = get_avg(cochlea1, 5, 5)
    cochlea4 = get_avg(cochlea1, 11, 11)
    
    all_cochleas = np.concatenate([cochlea1,cochlea2,cochlea3,cochlea4],0)
    del0 = deltas(all_cochleas)
    ddel = deltas(deltas(all_cochleas, 5), 5)

    ouotput = np.concatenate((all_cochleas, del0, ddel), 0)

    return ouotput

def gammatone(insig, numChan=128, fs = 16000): # 
    fRange = [1000, 20000] # try from 1000 to 20000 (was [50, 8000])
    filterOrder = 4
    gL = 2048
    sigLength = len(insig)
    phase = np.zeros([numChan, 1])
    erb_b = hz2erb(fRange)

    
    erb_b_diff = (erb_b[1]-erb_b[0])/(numChan-1)
    erb = np.arange(erb_b[0], erb_b[1]+epsc, erb_b_diff)
    cf = erb2hz(erb)
    b = [1.019 * 24.7 * (4.37 * x / 1000 + 1) for x in cf]
    gt = np.zeros([numChan, gL])
    tmp_t = np.arange(1,gL+1)/fs
    for i in range(numChan):
        gain = 10**((loudness(cf[i])-60)/20)/3*(2 * np.pi * b[i] / fs)**4
        tmp_temp = [gain*(fs**3)*x**(filterOrder - 1)*np.exp(-2 * np.pi * b[i] * x)*np.cos(2 * np.pi * cf[i] * x + phase[i]) for x in tmp_t]
        tmp_temp2 = np.reshape(tmp_temp, [1, gL])

        gt[i, :] = tmp_temp2

    sig = np.reshape(insig,[sigLength,1])
    gt2 = np.transpose(gt)
    resig = np.matlib.repmat(sig,1,numChan)
    r = np.transpose(fftfilt(gt2,resig,numChan))
    return r

def hz2erb(hz):  
    erb1 = 0.00437
    erb2 = np.multiply(erb1,hz)
    erb3 = np.subtract(erb2,-1)
    erb4 = np.log10(erb3)
    erb = 21.4 *erb4
    return erb

def erb2hz(erb): 
    hz = [(10**(x/21.4)-1)/(0.00437) for x in erb]
    return hz

def loudness(freq): 
    dB=60
    fmat = sio.loadmat('/content/gdrive/MyDrive/f_af_bf_cf.mat')
    af = fmat['af'][0]
    bf = fmat['bf'][0]
    cf = fmat['cf'][0]
    ff = fmat['ff'][0]
    i = 0
    while ff[i] < freq and i < len(ff) - 1: # my code:  i < len(ff)
        i = i + 1

    afy = af[i - 1] + (freq - ff[i - 1]) * (af[i] - af[i - 1]) / (ff[i] - ff[i - 1])
    bfy = bf[i - 1] + (freq - ff[i - 1]) * (bf[i] - bf[i - 1]) / (ff[i] - ff[i - 1])
    cfy = cf[i - 1] + (freq - ff[i - 1]) * (cf[i] - cf[i - 1]) / (ff[i] - ff[i - 1])
    loud = 4.2 + afy * (dB - cfy) / (1 + bfy * (dB - cfy))
    return loud

def fftfilt(b,x,nfft): 
    fftflops = [18, 59, 138, 303, 660, 1441, 3150, 6875, 14952, 32373, 69762,
                149647, 319644, 680105, 1441974, 3047619, 6422736, 13500637, 28311786,
                59244791, 59244791*2.09]
    nb, _ = np.shape(b)
    nx, mx = np.shape(x)
    n_min = 0
    while 2**n_min < nb-1:
        n_min = n_min+1
    n_temp = np.arange(n_min, 21 + epsc, 1)
    n = np.power(2,n_temp)
    fftflops = fftflops[n_min-1:21]
    L = np.subtract(n,nb-1)
    lenL= np.size(L)
    temp_ind0 = np.ceil(np.divide(nx,L))
    temp_ind = np.multiply(temp_ind0,fftflops)
    temp_ind = np.array(temp_ind)
    ind = np.argmin(temp_ind)
    nfft=int(n[ind])
    L=int(L[ind])
    b_tr = np.transpose(b)
    B_tr = fft(b_tr,nfft)
    B = np.transpose(B_tr)
    y = np.zeros([nx, mx])
    istart = 0
    while istart < nx :
        iend = min(istart+L,nx)
        if (iend - istart) == 1 :
            X = x[0][0]*np.ones([nx,mx])
        else :
            xtr = np.transpose(x[istart:iend][:])
            Xtr = fft(xtr,nfft)
            X = np.transpose(Xtr)
        temp_Y = np.transpose(np.multiply(B,X))
        Ytr = ifft(temp_Y,nfft)
        Y = np.transpose(Ytr)
        yend = np.min([nx, istart + nfft])
        y[istart:yend][:] = y[istart:yend][:] + np.real(Y[0:yend-istart][:])

        istart = istart + L
    
    return y

def cochleagram(r, winLength = 320, winShift=160): 
    numChan, sigLength = np.shape(r)
    increment = winLength / winShift
    M = np.floor(sigLength / winShift)
    a = np.zeros([numChan, int(M)])
    rs = np.square(r)
    rsl = np.concatenate((np.zeros([numChan,winLength-winShift]),rs),1)
    for m in range(int(M)):
        temp = rsl[:,m*winShift : m*winShift+winLength]
        a[:, m] = np.sum(temp,1)

    return a

def get_avg( m , v_span, h_span): 
    nr,nc = np.shape(m)

    fil_size = (2 * v_span + 1) * (2 * h_span + 1)
    meanfil = np.ones([1+2*h_span,1+2*h_span])
    meanfil = np.divide(meanfil,fil_size)

    out = signal.convolve2d(m, meanfil, boundary='fill', fillvalue=0, mode='same')
    return out

def deltas(x, w=9) : 
    nr,nc = np.shape(x)
    if nc ==0 :
        d= x
    else :
        hlen = int(np.floor(w / 2))
        w = 2 * hlen + 1
        win=np.arange(hlen, int(-(hlen+1)), -1)
        temp = x[:, 0]
        fx = np.matlib.repmat(temp.reshape([-1,1]), 1, int(hlen))
        temp = x[:, nc-1]
        ex = np.matlib.repmat(temp.reshape([-1,1]), 1, int(hlen))
        xx = np.concatenate((fx, x, ex),1)
        d = lfilter(win, 1, xx, 1)
        d = d[:,2*hlen:nc+2*hlen]

    return d



################# Mel Spectrogram Code

def get_mel_spec_from_file(file):
  mid = 2000000 # truncating or cutting the waveform tensors are less or more length
  waveform, sr = torchaudio.load(file)
  top_db = 80
  if len(waveform[0]) < mid:
    target = torch.zeros(mid) + 0.0005
    source = waveform[0]
    target[:len(source)] = source 
    spec = transform(target)
    return transforms.AmplitudeToDB(top_db=top_db)(spec)
  waveform = waveform[0]
  spec = transform(waveform[:mid])
  return transforms.AmplitudeToDB(top_db=top_db)(spec)