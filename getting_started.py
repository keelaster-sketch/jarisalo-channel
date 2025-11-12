#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel_torch getting started.py
Created on Wed Nov  5 20:50:50 2025
"""

import torch
import channel_torch as ch

# 3GPP Typical Urban SISO channel, 20 taps
def test_TUx(T=1,l_inp=128,nsps=1,spts=1.0,device='cpu'):
    'T      - number of frames to generate (def: 1)'
    'l_inp  - number of transmit symbols per frame (def: 128)'
    'nsps   - number of samples per tx symbol (def: 1)'
    'spts   - number of samples per Ts (def: 1.0)'
    'out    - complex channel output. TU model, no AWGN.'
    print(f'3GPP Typical Urban, {device}: frames={T}, # tx symbols={l_inp},samples per tx symbol={nsps}, samples per Ts={spts}, Ts=32.5ns')
    inp_bits=torch.randint(low=0,high=2,size=(T,l_inp),device=device)
    inp=ch.ch_inp_BPSK2(inp_bits,nsps=nsps,device=device) # includes tx filter
    CIR_mat,delays= ch.TUx(T=T,spts=spts,device=device)
    out=ch.ch_fft(inp,CIR_mat,delays,device=device)
    return out


import numpy as np
import matplotlib.pyplot as plt
nsps=8
l_inp=256
channel_output=test_TUx(l_inp=l_inp,nsps=nsps)    # send l_inp random symbols
channel_output=channel_output.numpy(force=True)
plt.plot(np.abs(channel_output.T))
#plt.plot(20*np.log10(np.abs(channel_output.T)))          
#plt.ylim((-40,10))
plt.xlabel('delay samples')
plt.ylabel('channel output, abs value')
plt.title(f'TU channel output, {l_inp} tx symbols, nsps={nsps}')
plt.show()

nsps=8      
l_inp=1
channel_output=test_TUx(l_inp=l_inp,nsps=nsps,spts=1.0)    # send l_inp=1 random symbol
channel_output=channel_output.numpy(force=True)
plt.stem(np.abs(channel_output.T))          # oversampled channel impulse response
plt.xlabel('delay samples')
plt.ylabel('channel output, abs value')
plt.title(f'TU CIR, nsps={nsps} samples per symbol')
plt.show()

nsps=1
l_inp=1
channel_output=test_TUx(l_inp=l_inp,nsps=nsps,spts=1.0)    # send l_inp=1 random symbol
channel_output=channel_output.numpy(force=True)
plt.stem(np.abs(channel_output.T))          # channel impulse response
plt.xlabel('delay samples')
plt.ylabel('channel output, abs value')
plt.title(f'TU CIR, nsps={nsps} samples per symbol')
plt.show()


