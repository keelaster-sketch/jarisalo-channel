#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 20:12:46 2025
Minimal pytorch AI/ML receiver training example, using channel_torch
"""
import torch
import torch.nn as nn
import torch.optim as optim
import channel_torch as ch

# Dataset generation
def ds_UniPdp(T=1, l_inp=128, fD=3.0, nsps=1, tauMax_ns=100, spts=1.0, device='cpu'):
    inp_syms = torch.randn((T, l_inp), device=device).sign()  # random BPSK
    inp = ch.ch_inp_sinc(inp_syms, nsps=nsps, device=device)
    CIR_mat, delays = ch.UniPdp(T=T, spts=spts, fD=fD, tauMax_ns=tauMax_ns, device=device)
    out = ch.ch_fft(inp, CIR_mat, delays, device=device)
    print(f'[Dataset] Tx frames={T},symbols/frame={l_inp},nsps={nsps}, out pwr={out.abs().pow(2).mean():.2e}')
    return out, inp_syms

# Complex loss
class ComplexMSELoss(nn.Module):
    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(inp - target) ** 2)

# Complex FIR model
class ComplexFIR(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.fir = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            dtype=torch.cfloat )

    def forward(self, x):
        return self.fir(x)


# EXAMPLE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate dataset (no noise, static channel, uniform PDP)
T, K = 4096, 128    # frames, symbols per frame
tauMax_ns=100       
rx_iq, tx_sym = ds_UniPdp(T=T, l_inp=K, fD=0.0, tauMax_ns=tauMax_ns, device=device)

# AI/ML model 
model = ComplexFIR(kernel_size=9).to(device)    # kernel size depends on tauMax
criterion = ComplexMSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-1)  # lr high since noiseless input

# Match shapes to (T,1,K) for the model
rx_iq = rx_iq[:, :K].unsqueeze(1)       # choose rx iq samples (time sync)
tx_sym = tx_sym.unsqueeze(1)

# training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    rx_sym_hat = model(rx_iq)
    loss = criterion(rx_sym_hat, tx_sym)
    loss.backward()
    optimizer.step()
    
    if epoch%10==0:
        print(f"[Epoch {epoch+1:03d}/{num_epochs}]  MSE Loss: {loss.item():.6e}")

# BER calcu;ation using the trained model, no AWGN
with torch.no_grad():
    # forward pass
    y = model(rx_iq)
    # decision: BPSK → sign(real part)
    decisions = torch.sign(y.real)
    # compare to true symbols (already ±1)
    bit_errors = (decisions != tx_sym).sum().item()
    ber = bit_errors / tx_sym.numel()

inp_pow=rx_iq.abs().pow(2).mean()
out_pow=y.abs().pow(2).mean()
print(f"[Post-training] inp/out pwr={inp_pow:.1e}/{out_pow:.1e}, MSE={loss.item():.2e}, BER={ber:.2e}")
