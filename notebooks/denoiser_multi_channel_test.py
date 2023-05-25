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
from one.api import ONE
import spikeinterface.preprocessing as si
import spikeinterface.extractors as se
import spikeinterface.full as sf
from spike_psvae import subtract
from pathlib import Path
import subprocess
import fileinput
import numpy as np
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas

# %%
one = ONE(base_url="https://alyx.internationalbrainlab.org")

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

# %%
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
import h5py
import re

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'

# template_raw_wfs_benchmark = dict()

for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    
    check_dir = main_dir + '/' + 'eID_' + eID + '_probe_' + probe + '_pID_' + pID
    ks_cluster_temp_dir = check_dir + '/ks_cluster_templates.npz'
    
    destriped_cbin_dir = list(Path(check_dir).glob('destriped_*.cbin'))[0]
    
    rec_cbin = sf.read_cbin_ibl(Path(check_dir))
    destriped_cbin = check_dir + '/' + destriped_cbin_dir.name
    #rec_cbin.get_num_channels()
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    h5_dir = check_dir + '/subtraction.h5'
    with h5py.File(h5_dir) as h5:
        channel_index = h5["channel_index"][:]
        

    ks_cluster_templates = np.load(ks_cluster_temp_dir)
    
    aligned_spike_train = ks_cluster_templates['aligned_spike_train']
    order = ks_cluster_templates['order']
    templates_aligned = ks_cluster_templates['templates_aligned']
    
    
    all_temps = list(Path(manually_picked_temp_dir).glob(pID + '*'))
    
    st = aligned_spike_train[order,:]
    # cluster_ids = np.unique(aligned_spike_train[:,1])
    
    # template_raw_wfs_benchmark[pID] = dict()
    # cluster_wfs = dict()
    cluster_temp = dict()
    max_channels = []
    
    for j in range(len(all_temps)):
        file_name = all_temps[j].name
        underline_idx = list(re.finditer('_', file_name))
        clu_id = int(file_name[underline_idx[1].end():underline_idx[2].start()])

        template = templates_aligned[clu_id]
            
        ptps = template.ptp(0)
        mc = np.argmax(ptps)
        max_channels.append(mc)
        
        ci = channel_index[mc]
        
        # cluster_wfs[clu_id] = []
        
        spks_in_cluster = np.squeeze(np.where(st[:,1] == clu_id))
        rand_sample_idx = np.random.choice(len(spks_in_cluster), 100)
        ci_not_nan = np.squeeze(np.where(ci < 384))
        # for k in range(100):
#             load_start = np.int32(st[spks_in_cluster][rand_sample_idx[k], 0] - 42)
#             load_end = np.int32(st[spks_in_cluster][rand_sample_idx[k], 0] + 79)

#             raw_wfs = np.zeros((121, 40)) * np.nan
            


#             raw_wfs[:,ci_not_nan] = rec.get_traces(start_frame=load_start, end_frame=load_end)[:,ci[ci_not_nan]]
            
#             cluster_wfs[clu_id].append(raw_wfs)
            
        temp_ci = np.zeros((121, 40)) * np.nan
        temp_ci[:,ci_not_nan] = template[:,ci[ci_not_nan]]
        cluster_temp[clu_id] = temp_ci#np.nanmean(np.array(cluster_wfs[clu_id]), axis = 0)
        
    template_raw_wfs_benchmark[pID]['temp'] = cluster_temp
    template_raw_wfs_benchmark[pID]['maxchan'] = max_channels
    # template_raw_wfs_benchmark[pID]['wfs'] = cluster_wfs
    # good_clusters = list(cluster_wfs.keys())
        
    

# %%
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'
template_raw_wfs_benchmark = np.load(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy')

# %%
template_raw_wfs_benchmark = template_raw_wfs_benchmark.item()

# %%
np.save(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy', template_raw_wfs_benchmark)

# %%
import matplotlib.pyplot as plt
i = 20
plt.plot(np.array(template_raw_wfs_benchmark['1a276285-8b0e-4cc9-9f0a-a3a002978724']['wfs'][10])[:,:,i].T, c = [0.5, 0.5, 0.5])
plt.plot(np.array(template_raw_wfs_benchmark['1a276285-8b0e-4cc9-9f0a-a3a002978724']['temp'][10])[:,i], c = 'r')

# %%
plt.plot(np.array(template_raw_wfs_benchmark['1a276285-8b0e-4cc9-9f0a-a3a002978724']['temp'][37]));

# %%
with h5py.File(h5_dir) as h5:
    geom = h5["geom"][:]
        

# %%
import torch

from spike_psvae import denoise

dn = denoise.SingleChanDenoiser().load()
device = None

waveforms = np.array(template_raw_wfs_benchmark['1a276285-8b0e-4cc9-9f0a-a3a002978724']['wfs'][37])
template =  np.array(template_raw_wfs_benchmark['1a276285-8b0e-4cc9-9f0a-a3a002978724']['temp'][37])

ptps = template.ptp(1)
maxchan = np.nanargmax(ptps)
maxchans = np.ones(np.shape(waveforms)[0])*maxchan

waveforms = torch.as_tensor(waveforms, device=device, dtype=torch.float)
waveforms = denoise.multichan_phase_shift_denoise(waveforms, geom, channel_index, dn, maxchans = maxchans)
waveforms = torch.as_tensor(waveforms, device=device, dtype=torch.float)


waveforms = waveforms.permute(0, 2, 1)

# %%
j

# %%
bias = bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
offset= 10

save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/multi-channel_denoise_compare/'
for i in range(0,1):#len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    
    units = list(template_raw_wfs_benchmark[pID]['temp'].keys())
    
    for j in range(18, len(units)):
        
        raw_wfs = np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])
        template =  np.array(template_raw_wfs_benchmark[pID]['temp'][units[j]])

        ptps = template.ptp(1)
        maxchan = np.nanargmax(ptps)
        maxchans = np.ones(np.shape(raw_wfs)[0])*maxchan

        waveforms = torch.as_tensor(raw_wfs, device=device, dtype=torch.float)
        waveforms = denoise.multichan_phase_shift_denoise(waveforms, geom, channel_index, dn, maxchans = maxchans)
        
        
        
        wfs = np.swapaxes(raw_wfs, 1, 2)
        wfs_denoised_old = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
        wfs_denoised_old = wfs_denoised_old.detach().numpy()
        wfs_denoised_old = np.swapaxes(wfs_denoised_old, 1, 2)
        
        for k in range(np.shape(raw_wfs)[0]):

            fig, axs = plt.subplots(1, 4, figsize = [12, 6])

            axs[0].plot(raw_wfs[k] + bias*offset, c = 'k')
            axs[0].plot(waveforms[k] + bias*offset, c = 'g')
            axs[0].set_title('raw vs phase-shift denoise')

            axs[1].plot(raw_wfs[k] + bias*offset, c = 'k')
            axs[1].plot(wfs_denoised_old[k] + bias*offset, c = 'b')
            axs[1].set_title('raw vs singleCH denoise')

            axs[2].plot(template + bias*offset, c = 'r')
            axs[2].plot(waveforms[k] + bias*offset, c = 'g')
            axs[2].set_title('temp vs phase-shift denoise')

            axs[3].plot(template + bias*offset, c = 'r')
            axs[3].plot(wfs_denoised_old[k] + bias*offset, c = 'b')
            axs[3].set_title('temp vs singleCH denoise')
            
            plt.savefig(save_dir + pID + '_unit_' + str(j) + '_' + str(k) + '.png', dpi = 150)
            
            plt.close()

# %%
