#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample torch dataset creation for SISO channel simulation with channel_torch.

Notation:
T       - number of CIRs to generate. Positive integer. 
fD      - max Doppler frequency, Hz. Positive float.
t_step  - time interval between CIRs, seconds. Positive float.
l_inp   - number of tx symbols, BPSK or QPSK. Positive integer.
nsps    - number of samples per transmit symbol. Positive integer.
"""
# CIR=Channel Impulse Response

import torch
import channel_torch as ch
from torch.utils.data import Dataset
import os

# custom pytorch dataset
class DatasetExample(Dataset):
    def __init__(self):
        self.ch_out = [] # complex
        self.inp_bits = [] # integer 

    def __len__(self):
        return len(self.ch_out)
    
    def __getitem__(self, index):
        x1 = self.ch_out[index]
        x2 = self.inp_bits[index]
        return x1, x2
    
    def add_item(self, new_ch_out, new_inp_bits):
        self.ch_out.append(new_ch_out) 
        self.inp_bits.append(new_inp_bits) 
dataset = DatasetExample()  # complex channel output, input bits

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
    return out, inp_bits


T=1000          # num transmitted frames
fD=3.0          # max Doppler in Hz 
t_step=1.0      # time interval between CIRs, seconds
l_inp=128       # num transmitted symbols in frame
nsps=1          # positive integer transmitter oversampling factor

# Create Typical Urban BPSK dataset
ch_out,inp_bits = test_TUx(T=T,l_inp=l_inp,nsps=nsps,spts=1.0,device='cpu')

for t in torch.arange(0,T):
    if t % 100000== 0:
        print(f'Adding t={t}/{T}')
    dataset.add_item(ch_out[t,:],inp_bits[t,:]) 

print(f'TUx dataset created, {t+1} records. ')
# sanity check
# channel output mean sample power and total power 
print(f'ch_out mean sample power={torch.mean(torch.abs(ch_out)**2)}')
print(f'ch_out sum power={torch.sum(torch.abs(ch_out)**2)}')

print('Saving on disk..')
ds_fn='SAMPLE_DATASET.pth'
torch.save(dataset, ds_fn) # CHECK YOUR PATH
ds_mb=round(os.path.getsize(ds_fn)/(1024*1024),ndigits=1) 
print(f'TUx dataset saved ({ds_mb} MB): T={T}, num tx symbols={l_inp}, nsps={nsps}, ch_out length={ch_out.shape[1]}, Ts=32.55ns')

# loading dataset
dataset_loaded = torch.load(ds_fn, weights_only=False)





