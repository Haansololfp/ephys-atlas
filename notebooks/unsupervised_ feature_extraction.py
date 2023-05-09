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

# %% [markdown]
# Try unsupervised feature extraction on spike waveforms:
# -PCA
# -UMAP
# -VAE

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from one.api import ONE
from ibllib.atlas import BrainRegions
import pandas as pd
from pathlib import Path
import h5py
from scipy import signal
import os

# %%
Benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724', 
                  '1e104bf4-7a24-4624-a5b2-c2c8289c0de7', 
                  '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e', 
                  '5f7766ce-8e2e-410c-9195-6bf089fea4fd', 
                  '6638cfb3-3831-4fc2-9327-194b76cf22e1', 
                  '749cb2b7-e57e-4453-a794-f6230e4d0226', 
                  'd7ec0892-0a6c-4f4f-9d8f-72083692af5c', 
                  'da8dfec1-d265-44e8-84ce-6ae9c109b8bd', 
                  'dab512bd-a02d-4c1f-8dbc-9155a163efc0', 
                  'dc7e9403-19f7-409f-9240-05ee57cb7aea', 
                  'e8f9fba4-d151-4b00-bee7-447f0f3e752c', 
                  'eebcaf65-7fa4-4118-869d-a084e84530e2', 
                  'fe380793-8035-414e-b000-09bfe5ece92a']
one = ONE(base_url="https://alyx.internationalbrainlab.org")
br = BrainRegions()


# %% [markdown]
# # align the waveforms to the maxCH

# %%
def align_spikes(wfs):
# input: waveforms of shape NxTxC
# upsample the waveform in time first for finer scale alignment
# then downsample the waveform to its original shape
    N, T, C = np.shape(wfs)
    upsampled_wfs = signal.resample(waveforms, 10*T, axis = 1)
    aligned_wfs = np.zeros([N, 1400, C + 1])
    
    peak_idx = 500 # a value that is larger than all peak index values
    
    wfs_ptps = upsampled_wfs.ptp(1)
    mcs = np.nanargmax(wfs_ptps, axis =1)
    
    
    peak_point = np.argmax(np.absolute(upsampled_wfs[np.arange(N), :, mcs]), axis =1)
    
    shift = peak_idx - peak_point
    
    discard_idx = np.where((mcs!=19) & (mcs!=20))
    
    shift_ch_idx = np.squeeze(np.where(mcs == 19)) # if max channel is 19, shift the waveform by one channel
    non_shift_ch_idx = np.squeeze(np.where(mcs == 20))  #if max channel is 20, keep the current waveform 

    
    for i in range(len(shift_ch_idx)):
        idx = shift_ch_idx[i]
        aligned_wfs[idx, shift[idx]:(shift[idx] + 10*N), 1:None] = upsampled_wfs[idx,:,:]
    for i in range(len(non_shift_ch_idx)):
        idx = non_shift_ch_idx[i]
        aligned_wfs[idx, shift[idx]:(shift[idx] + 10*N), 0:None-1] = upsampled_wfs[idx,:,:]
        
    return discard_idx, aligned_wfs

# %%
i = 0
pID =Benchmark_pids[i]
eID, probe = one.pid2eid(pID)

main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
h5_path = out_dir + '/' + 'subtraction.h5'

with h5py.File(h5_path) as h5:
    waveforms = h5['denoised_waveforms'][0:10000]

# %%
discard_idx, aligned_wfs = align_spikes(waveforms)

# %% [markdown]
# - uniformly sample (# sample balanced subsets from different berain regions)
# - normalize waveform
# - raw waveform + denoised waveform

# %%
example_snippet_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example/eebcaf65-7fa4-4118-869d-a084e84530e2'

# %%
from pathlib import Path
p = Path(example_snippet_dir)
snippets_dir = list(p.glob('T*'))

# %%
snippets_dir[0].joinpath('spikes.pqt')
df_spikes = pd.read_parquet(snippets_dir[j].joinpath('spikes.pqt'))

# %%

main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
i = 0
pID = Benchmark_pids[i]
eID, probe = one.pid2eid(pID)

LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))


df_channels = df_channels.reset_index(drop=False)
df_channels = df_channels[df_channels.pid == pID]
df_channels = df_channels.reset_index(drop=True)

channel_ids = df_channels['atlas_id'].values


br = BrainRegions()
region_info = br.get(channel_ids)
boundaries = np.where(np.diff(region_info.id) != 0)[0]
boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]

regions = np.c_[boundaries[0:-1], boundaries[1:]]

channel_depths=df_channels['axial_um'].values
if channel_depths is not None:
    regions = channel_depths[regions]
region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]] #depth + region_labels

out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
h5_path = out_dir + '/' + 'subtraction.h5'

with h5py.File(h5_path) as h5:
    z = h5['localizations'][:,2]

bins = np.unique(regions)
inds = np.digitize(z, bins,right=True)

c, bins = np.histogram(inds)

p = c[1:None-1]/ np.sum(c[1:None-1])



# %%
bins = np.unique(regions)
inds = np.digitize(z, bins,right=True)

# %%
c, bins = np.histogram(inds)

# %%
uni_idxs

# %%
np.shape(np.where(inds == uni_idxs[i]))

# %%
for i in range(1, len(uni_idxs)-1, 1):
    # if np.shape(np.where(inds == uni_idxs[i]))[1]>1000:
    print(np.shape(np.where(inds == uni_idxs[i]))[1])
    plt.hist(z[inds == uni_idxs[i]], bins = np.arange(regions[i-1][0], regions[i-1][1]))

# %%
regions[:,1] - regions[:,0]

# %%
a = [7308,
     620,
     52612,
     3018,
     72814,
     1540,
     3636,
     7361,
     120163,
     51277,
     6581,
     5823]
plt.bar(np.arange(len(a)),(regions[:,1] - regions[:,0])/a)

# %%
import spike_feature_VAE

# %%
import importlib
importlib.reload(spike_feature_VAE)


# %%
def normalize_waveform_by_peak(wfs):
    peak_abs_value = np.abs(wfs).nanmax(1).nanmax(1)
    print(peak_abs_value)
    return np.divide(wfs, peak_abs_value[:,None, None], )


# %%
with h5py.File(h5_path) as h5:
    waveforms = h5["denoised_waveforms"][:]


# %%
#load denoised waveforms equal number in each region
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")
wfs_all = dict()
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    h5_path =out_dir + '/' + 'subtraction.h5'
    
    with h5py.File(h5_path) as h5:
        z = h5['localizations'][:,2]
        max_ptps = h5['maxptps'][:]
        spike_times = h5["spike_index"][:,0]/30000
    
    which = slice(None)
        
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)
    
    channel_ids = df_channels['atlas_id'].values

    region_info = br.get(channel_ids)
    boundaries = np.where(np.diff(region_info.id) != 0)[0]
    boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]

    regions = np.c_[boundaries[0:-1], boundaries[1:]]

    channel_depths=df_channels['axial_um'].values

    regions = channel_depths[regions]
    region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]
    
    z = z[which]
    
    #uniformly sample from BRs
    bins = np.unique(regions)
    inds = np.digitize(z, bins,right=True)
    inds = inds - 1 
    
    for j in range(len(region_labels)):
        which_region = np.where(inds == j)
        if (np.shape(which_region)[1]>1000):
            np.random.seed(0)
            subsample = np.random.choice(np.squeeze(which_region), 1000, replace=False)
            
            with h5py.File(h5_path) as h5:
                waveforms = h5["denoised_waveforms"][np.sort(subsample),:,:]

            RL = region_labels[j,1]
            if RL in wfs_all.keys():
                wfs_all[RL] = np.append(wfs_all[RL], waveforms, axis = 0)
            else:
                wfs_all[RL] = waveforms
            

# %%
brain_regions_list = wfs_all.keys()

# %%
data = np.concatenate([wfs_all[x] for x in brain_regions_list], 0)

# %%
normalized_data = normalize_waveform_by_peak(data)

# %%
np.shape(data)

# %%
N_C_nan_idx = np.where(np.isnan(data))
data[N_C_nan_idx] = 0

# %%
plt.plot(normalized_data[1,:,:]);

# %%
np.shape(data)

# %%
data = np.expand_dims(normalized_data, axis = 1)

# %%
train_data, test_data = torch.utils.data.random_split(data, [round(0.7*3796), round(0.3*3796)], generator=torch.Generator().manual_seed(42))

# %%
import torch
from torch.utils.data import DataLoader

cuda = False
device = torch.device("cuda" if cuda else "cpu")

batch_size = 1000
latent_dims = 20

num_epochs = 200

torch.manual_seed(40)
model = spike_feature_VAE.Autoencoder(latent_dims=latent_dims,).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
LOSS = []


train_loader = DataLoader(dataset=data, batch_size=1000)

for epoch in range(num_epochs):
    tmp_val_losses = []
    for idx, data_batch in enumerate(train_loader, 1):
        input_x = data_batch.to(device)

        reconstruct_x, mu, log_var, t = model(input_x)

        optimizer.zero_grad()
        elbo = spike_feature_VAE.build_loss(reconstruct_x, input_x, mu, log_var)
        tmp_val_losses.append(elbo.detach().numpy())
        
        elbo.backward()
        optimizer.step()
        # print('mu:' + str(mu) + ',var:' + str(log_var))
        print('epoch ' + str(epoch) + ', batch' + str(idx) + ' elbo:' + str(elbo.detach().numpy()))
    LOSS.append(sum(tmp_val_losses)/(idx+1))  
    print('epoch ' + str(epoch) + 'ELBO:' + str(sum(tmp_val_losses)/(idx+1)) + '\n')
    

# %%
