#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 18:07:36 2025
channel_torch usage example with MMSE equalizer with perfect CSI
"""
#CSI=Channel State Information

import torch
import torch.nn as nn
import numpy as np
import channel_torch as ch
import matplotlib.pyplot as plt

# data creation helper, uniform channel PDP 
def test_UniPdp(T=1,l_inp=128,nsps=1,spts=1.0,device='cpu'):
    'T      - number of frames to generate (def: 1)'
    'l_inp  - number of transmit symbols per frame (def: 128)'
    'nsps   - number of samples per tx symbol (def: 1)'
    'spts   - number of samples per Ts (def: 1.0)'
    'out    - complex channel output. TU model, no AWGN.'
    print(f'Device={device}, frames={T}, # tx symbols={l_inp},samples per tx symbol={nsps}, samples per Ts={spts}, Ts=32.5ns')
    inp_syms=torch.randn((T,l_inp),device=device).sign_() # random BPSK
    inp=ch.ch_inp_sinc(inp_syms,nsps=nsps,device=device) # includes tx filter
    CIR_mat,delays= ch.UniPdp(T=T,spts=spts,device=device)
    out=ch.ch_fft(inp,CIR_mat,delays,device=device)
    return out,inp_syms


def next_pow2(n):
    return 1 << ((n - 1).bit_length())


class MMSEEqualizer(nn.Module):
    """
    Frequency-domain MMSE equalizer that supports oversampling (nsps>1).

    forward(y, h, noise_var=1e-3, sig_pow=1.0, nsps=1, samp_offset=0, K_sym=None)
    - y: (B, N_samples) complex received waveform (sample rate)
    - h: (B, L_ir) complex impulse responses aligned to sample grid
    - noise_var: scalar or (B,) noise variance per complex sample
    - sig_pow: average transmit power per (complex) sample (scalar)
    - nsps: samples per symbol (int >=1)
    - samp_offset: integer in [0, nsps-1] selecting which sample is the symbol instant
    - K_sym: optional number of symbols to return per batch (if None infer from y)
    Returns:
    - x_hat_sym: (B, K_sym) complex tensor of symbol-rate equalized samples
    - (optionally) x_hat_time: full time-domain equalized samples if you need them
    """
    def __init__(self):
        super().__init__()

    def forward(self, y, h, noise_var=1e-3, sig_pow=1.0, nsps=1, samp_offset=0, K_sym=None, return_time=False):
        # basic checks / cast
        if not torch.is_complex(y) or not torch.is_complex(h):
            raise ValueError("y and h must be complex tensors")

        B, N = y.shape
        _, L = h.shape
        nsps = int(nsps)
        samp_offset = int(samp_offset) % max(nsps, 1)
        if nsps < 1:
            nsps = 1
        # FFT length consistent with linear convolution
        fft_len = next_pow2(N + L - 1)

        # FFTs
        Yf = torch.fft.fft(y, n=fft_len)         # (B, fft_len)
        Hf = torch.fft.fft(h, n=fft_len)         # (B, fft_len)

        # noise_var handling (broadcast to batch)
        noise_var = torch.as_tensor(noise_var, dtype=torch.float, device=y.device)
        if noise_var.dim() == 0:
            noise_var = noise_var.expand(B)
        elif noise_var.numel() == 1 and B > 1:
            noise_var = noise_var.repeat(B)

        # denom per-freq per-batch
        denom = (torch.abs(Hf) ** 2) + (noise_var.view(B, 1) / float(sig_pow)) + 1e-12
        Wf = torch.conj(Hf) / denom               # (B, fft_len)

        # equalize in freq domain
        Xhat_f = Wf * Yf
        x_hat_time = torch.fft.ifft(Xhat_f, n=fft_len)   # (B, fft_len)

        # Trim to original time length N (linear conv produced at least N samples)
        x_hat_time = x_hat_time[:, :N]   # (B, N)

        if nsps==1: 
            total_offset=0
        else:   # tx sinc filter
            total_offset=(3*nsps+samp_offset+1)
        
        # Downsample to symbol instants
        x_hat_sym = x_hat_time[:, total_offset::nsps][:, :K_sym]  # (B, K_sym)

        if return_time:
            return x_hat_sym, x_hat_time
        else:
            return x_hat_time, x_hat_sym 


# --- helper for MMSE BER for one SNR point, perfect CSI ---
def run_mmse_with_UniPdp(
    B=4,            # Nnumber of transmitted frames
    K=128,          # num tx symbols per frame
    nsps=1,         # samples per tx symbol
    fD=3.0,         # max Doppler, Hz
    spts=1.0,       # samples per Ts, Ts=32.55ns
    EbN0_dB=100.0,
    tauMax_ns=2140,     # matches with TUx
    device='cpu'
):
    # --- Generate bits & TX waveform ---
    bits = torch.randint(0,2, (B, K), device=device)
    inp_syms=2.0*bits-1.0   # BPSK
    # Channel input 
    x_time = ch.ch_inp_sinc(inp_syms, nsps=nsps, device=device).to(torch.cfloat)
    # Uniform channel PDP, max delay is tauMax_ns nanoseconds
    CIR_matrix,delay_vec=ch.UniPdp(T=B,spts=spts,tauMax_ns=tauMax_ns,device=device)
    # Channel output
    y = ch.ch_fft(x_time, CIR_matrix, delay_vec, device=device)
    # --- Add AWGN, signal power is measured over batch
    EbN0 = 10 ** (EbN0_dB / 10)
    sig_pow = x_time.abs().pow(2).mean() 
    noise_var = sig_pow / EbN0
    noise = torch.sqrt(noise_var / 2) * (
        torch.randn_like(y) + 1j * torch.randn_like(y) )
    y_noisy = y + noise

    # --- Format channel impulse response for equalizer sparse format
    tau_max_samples=delay_vec.max()-delay_vec.min() + 1  
    CIR_matrix2 = torch.zeros((B, tau_max_samples), dtype=torch.cfloat, device=device)
    CIR_matrix2[:, delay_vec] = CIR_matrix

    # Sampling time offset (assumed known)
    best_offset = 0

    # MMSE equalizer
    eq = MMSEEqualizer().to(device)
    x_hat_time, x_hat_sym = eq(    y_noisy, 
                                   CIR_matrix2,
                                   noise_var=noise_var, 
                                   sig_pow=sig_pow,
                                   nsps=nsps, 
                                   samp_offset=best_offset, 
                                   K_sym=K)

    # --- Bit decisions and BER ---
    decisions = (x_hat_sym.real >= 0).int()
    bit_errors = (decisions != bits).sum().item()
    ber = bit_errors / (B * K)

    print(f"[MMSE Oracle] B={B},K={K},tau_max={tau_max_samples-1},nsps={nsps},sig pow={sig_pow},Es/N0={EbN0_dB:.1f} dB, BER={ber:.2e},time_offset={best_offset}")

    return ber, x_hat_time, x_hat_sym


# theoretical BER for BPSK over flat-fading Rayleigh (perfect CSI)
def ber_theory_rayleigh_bpsk(ebn0_lin):
    # ebn0_lin: scalar or array
    ebn0 = np.array(ebn0_lin, dtype=float)
    return 0.5 * (1.0 - np.sqrt(ebn0 / (1.0 + ebn0)))


def ber_rayleigh_mrc_theory(ebn0_lin, L=1, n_theta=20000):
    """
    Compute BPSK average BER over i.i.d. Rayleigh branches combined with MRC (L branches).
    ebn0_lin : scalar or 1D array of average Eb/N0 (linear, not dB)
    L        : number of diversity branches (integer >=1)
    n_theta  : number of theta samples for numerical integration (increase for more accuracy)
    Returns: BER array same shape as ebn0_lin
    Formula: Pe = (1/pi) * integral_0^{pi/2} (1 + ebn0/(L * sin^2 theta))^{-L} dtheta
    """
    ebn0_vec = np.atleast_1d(np.array(ebn0_lin, dtype=float))
    thetas = np.linspace(0.0 + 1e-12, np.pi/2.0 - 1e-12, n_theta)   # avoid exact endpoints
    sin2 = np.sin(thetas)**2
    # shape (n_theta, 1) for broadcasting
    sin2 = sin2[:, None]
    eb = ebn0_vec[None, :]  # (1, N)
    # integrand: (1 + eb/(L * sin^2 theta))^{-L}
    integrand = (1.0 + eb / (L * sin2))**(-L)
    integral = np.trapz(integrand, thetas, axis=0)  # integrate over theta
    pe = (1.0 / np.pi) * integral
    return pe if pe.size > 1 else float(pe)



# run MMSE equalizer example with uniform PDP
EbN0_dB_list = np.arange(-10, 25, 2)   
ebn0_lin = 10 ** (EbN0_dB_list / 10.0)

T = 4096         # num frames
l_inp = 128      # num symbols per frame
nsps = 1         # symbol oversampling factor, positive integer
tauMax_ns=2132.0   # uniform channel PDP max delay, nanoseconds. Ts=32.55ns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device={device}')
# SNR loop
ber_sim = []
for EbN0_dB in EbN0_dB_list:
    ber,_,_ = run_mmse_with_UniPdp(B=T, 
                                   K=l_inp, 
                                   nsps=nsps, 
                                   EbN0_dB=EbN0_dB,
                                   tauMax_ns=tauMax_ns, 
                                   device=device) 
    ber_sim.append(ber)

# --- plot result, MMSE simulation versus MRC bound
plt.figure()
plt.semilogy(EbN0_dB_list, ber_sim, 'o-', label='Sim MMSE w/ perfect CSI')
for L in (1, 2, 4, 8):
    ber_th = ber_rayleigh_mrc_theory(ebn0_lin, L=L, n_theta=8000)
    plt.semilogy(EbN0_dB_list, ber_th, label=f'MRC bound L={L}')

plt.grid(True, which='both')
plt.xlabel('Eb/N0 [dB]')
plt.ylabel('BER')
plt.legend()
plt.title(f'BPSK over UniPdp — MMSE vs MRC bound, nsps={nsps}')
plt.tight_layout()
plt.show()    
    
 