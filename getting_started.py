#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel_torch usage examples and plots
Created on Wed Nov  5 20:50:50 2025
"""
# CIR=Channel Impulse Response
# SISO=Single-Input Single-Output
import torch
import channel_torch as ch

# 3GPP TR 25.943 Typical Urban SISO channel, 20 delay taps
def test_TUx(T=1,l_inp=128,nsps=1,spts=1.0,device='cpu'):
    'T      - number of frames to generate (def: 1)'
    'l_inp  - number of transmit symbols per frame (def: 128)'
    'nsps   - number of samples per tx symbol (def: 1)'
    'spts   - number of samples per Ts (def: 1.0)'
    'out    - complex channel output. TU model, no AWGN.'
    print(f'3GPP Typical Urban, {device}: frames={T}, # tx symbols={l_inp},samples per tx symbol={nsps}, samples per Ts={spts}, Ts=32.5ns')
    inp_syms=torch.randn((T,l_inp),device=device).sign_() # random BPSK
    inp=ch.ch_inp_sinc(inp_syms,nsps=nsps,device=device) # includes tx filter
    CIR_mat,delays= ch.TUx(T=T,spts=spts,device=device)
    out=ch.ch_fft(inp,CIR_mat,delays,device=device)
    return out

import numpy as np
import matplotlib.pyplot as plt

# plot channel output for different inputs
nsps=8
l_inp=256
channel_output=test_TUx(l_inp=l_inp,nsps=nsps)    # send l_inp random symbols
channel_output=channel_output.numpy(force=True)
plt.plot(np.abs(channel_output.T))
plt.xlabel('delay samples')
plt.ylabel('channel output, abs value')
plt.title(f'TU channel fading output, {l_inp} tx symbols, nsps={nsps}')
plt.show()

nsps=8      
l_inp=1
channel_output=test_TUx(l_inp=l_inp,nsps=nsps,spts=1.0)    # send l_inp=1 random symbol
channel_output=channel_output.numpy(force=True)
plt.stem(np.abs(channel_output.T))          # oversampled channel impulse response
plt.xlabel('delay samples')
plt.ylabel('channel output, abs value')
plt.title(f'TU impulse response, nsps={nsps} samples per symbol')
plt.show()

nsps=1
l_inp=1
channel_output=test_TUx(l_inp=l_inp,nsps=nsps,spts=1.0)    # send l_inp=1 random symbol
channel_output=channel_output.numpy(force=True)
plt.stem(np.abs(channel_output.T))          # channel impulse response
plt.xlabel('delay samples')
plt.ylabel('channel output, abs value')
plt.title(f'TU impulse response, nsps={nsps} samples per symbol')
plt.show()






