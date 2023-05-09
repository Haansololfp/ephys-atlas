# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from spike_psvae.denoise import SingleChanDenoiser
import torch
from one.api import ONE
from pathlib import Path
import spikeinterface.full as sf
import scipy
import h5py
import matplotlib.pyplot as plt
from one.remote import aws
import pandas as pd

# %%
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/phase-shift_single_channel_denoiser_large_scale_test'
data = np.load(save_dir + '/waveforms_to_denoise.npz')
maxchan = data['maxchan']
wfs_to_denoise = data['wfs_to_denoise']

# %%
dn = SingleChanDenoiser().load()
wfs = np.swapaxes(wfs_to_denoise, 1, 2)
wfs_denoised_old = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)

# %%
wfs_denoised_old = wfs_denoised_old.detach().numpy()

# %%
fig, axs = plt.subplots(1, 3, figsize = (30, 10))
A = np.dot(wfs_denoised_old[0,:,:], wfs[0,:,:].T)
B = np.dot(wfs_denoised_old[0,:,:], wfs_denoised_old[0,:,:].T)
axs[0].imshow(A, vmin = 0, vmax = 100)
axs[1].imshow(B, vmin = 0, vmax = 100)
axs[2].imshow(np.divide(A,B), vmin = 0, vmax = 10)
# plt.colorbar()

# %%
C = np.diag(np.divide(B,np.maximum(A, np.ones(B.shape)*1E-6)))

# %%
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/denoise_hallucination_fix'

# %%
bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)

for i in range(50):

    A = np.dot(wfs_denoised_old[i,:,:], wfs[i,:,:].T)
    B = np.dot(wfs_denoised_old[i,:,:], wfs_denoised_old[i,:,:].T)

    C = np.divide(np.diag(B),np.diag(np.maximum(A, np.zeros(B.shape))))
    

    fig, axs = plt.subplots(1,4, figsize = [20, 10])
    
    axs[0].plot(wfs[i,:,:].T + bias*4, 'k');
    axs[0].plot(wfs_denoised_old[i,:,:].T + bias*4, 'r');
    axs[0].set_title('single channel')
    
    axs[1].plot(wfs[i,:,:].T + bias*4, 'k');
    axs[1].plot(wfs_denoised_old[i,:,:].T/np.maximum(C, 1) + bias*4, 'r');
    axs[1].plot((wfs_denoised_old[i,:,:].T/np.maximum(C, 1) + bias*4)[:, C>1], 'g');
    axs[1].set_title('single channel w supression')
    
    axs[2].imshow(wfs_denoised_old[i,:,:], aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[2].set_title('single channel')
    axs[3].imshow((wfs_denoised_old[i,:,:].T/np.maximum(C, 1)).T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[3].set_title('single channel w supression')
    # axs[1].set_title('divide by min(c, 1)')
    # axs[2].plot(wfs_denoised_old[i,:,:].T/np.maximum(C, 1) + bias*2, 'k');
    # axs[2].set_title('divide by max(c, 1)')
    
    plt.savefig(save_dir + '/unit_' + str(i) + '_suppress.png')

# %%
