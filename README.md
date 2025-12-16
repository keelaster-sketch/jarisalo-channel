# channel_torch
Easy-to-use pytorch functions for wireless channel simulation. 
Native complex-valued computation with symbol oversampling. 
Supports CUDA. This version is SISO-only.

## Usage examples:
 - getting_started, plots CIRs of 3gpp Typical Urban model
 - MMSE equalizer with perfect CSI
 - AI/ML equalizer with training, with perfect CSI
 - AI/ML dataset creation

## Notation:
T       - number of CIRs to generate. Positive integer. 
fD      - max Doppler frequency, Hz. Positive float.
t_step  - time interval between CIRs, seconds. Positive float.
spts    - # channel samples per Ts. Positive float. Ts=32.55 nanoseconds
l_inp   - # complex tx symbols, e.g., BPSK or QPSK. Positive integer.
nsps    - # samples per transmit symbol. Positive integer.
L       - # channel taps. Positive integer.

## Abbreviations:
CIR=Channel Impulse Response
CSI=Channel State Information

