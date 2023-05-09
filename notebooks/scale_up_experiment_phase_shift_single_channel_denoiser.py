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
# main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example'

# %%
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/phase-shift_single_channel_denoiser_large_scale_test'

# %%
# waveforms_to_denoise = []
# max_chan = []
# for path in Path(main_dir).rglob('spikes.pqt'):
#     wfs = np.load(path)
#     spikes = pd.read_parquet(list(path.parents[0].glob('spikes.pqt'))[0])
#     mcs = spikes['trace'].values
#     n_spk = len(mcs)
#     rand_idx = np.random.choice(n_spk, size =100)
#     waveforms_to_denoise.append(wfs[rand_idx,:,:])
#     max_chan.append(mcs[rand_idx])

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'

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
wfs_to_denoise = []
maxchan = []
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    h5_path = out_dir + '/' + 'subtraction.h5'
    sample_size = 200
    fs = 30000
    
    with h5py.File(h5_path) as h5:
        spike_idx = h5["spike_index"][:]

    spk_times = spike_idx[:,0]
    spk_channels = spike_idx[:,1]
    
    destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    load_start = spk_times - 42
    load_end = spk_times + 79
    
    spk_idx = np.random.choice(len(spk_times), sample_size)
    
    for j in range(len(spk_idx)):
        idx = spk_idx[j]
        ci = channel_index[spk_channels[idx]]
        
        raw_wfs = np.zeros((121, 40)) * np.nan
        
        non_nan_idx = np.where(ci<384)[0]
        
        not_nan_wfs = rec.get_traces(start_frame=load_start[idx], end_frame=load_end[idx])[:,ci[non_nan_idx]]
        
        raw_wfs[:,non_nan_idx] = not_nan_wfs
        
        wfs_to_denoise.append(raw_wfs)
        maxchan.append(spk_channels[idx])
        
        
wfs_to_denoise = np.array(wfs_to_denoise)

# %%
np.savez(save_dir + '/waveforms_to_denoise.npz', maxchan = maxchan, wfs_to_denoise = wfs_to_denoise)

# %%
data = np.load(save_dir + '/waveforms_to_denoise.npz')
maxchan = data['maxchan']
wfs_to_denoise = data['wfs_to_denoise']

# %%
waveforms_to_denoise = np.concatenate(waveforms_to_denoise)
max_chan = np.array(max_chan).flatten()

# %%
h5_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_111c1762-7908-47e0-9f40-2f2ee55b6505_probe_probe01_pID_eebcaf65-7fa4-4118-869d-a084e84530e2/subtraction.h5'

# %%
with h5py.File(h5_dir) as h5:
    geom = h5['geom'][:]
    channel_index = h5['channel_index'][:]


# %%
def denoise_with_phase_shift_w_suppress(chan_wfs, phase_shift, chan_ci_idx, spk_sign):

    wfs_to_denoise = np.roll(chan_wfs, -phase_shift)
    wfs = np.swapaxes(wfs_to_denoise[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = wfs_denoised.detach().numpy()
    
    
    A = np.dot(np.squeeze(wfs_denoised).T, wfs_to_denoise)
    B = np.dot(np.squeeze(wfs_denoised).T,np.squeeze(wfs_denoised))
    

    C = B/max(A, 0)
    
    if C > 1:
        SC.append(chan_ci_idx)
    spk_denoised_wfs[:, chan_ci_idx] = np.roll(np.squeeze(wfs_denoised), phase_shift)/max(C, 1)
    phase_shifted = np.argmax(wfs_denoised * spk_sign) - 42 + phase_shift
    
    return phase_shifted


# %%
def denoise_with_phase_shift(chan_wfs, phase_shift, chan_ci_idx, spk_sign):

    wfs_to_denoise = np.roll(chan_wfs, -phase_shift)
    wfs = np.swapaxes(wfs_to_denoise[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = wfs_denoised.detach().numpy()
    
    
    spk_denoised_wfs[:, chan_ci_idx] = np.roll(np.squeeze(wfs_denoised), phase_shift)
    phase_shifted = np.argmax(wfs_denoised * spk_sign) - 42 + phase_shift
    # phase_shifted = np.argmax(np.abs(wfs_denoised)) - 42 + phase_shift
    return phase_shifted

# %%
x_pitch = np.diff(np.unique(geom[:,0]))[0]
y_pitch = np.diff(np.unique(geom[:,1]))[0]

# %%
dn = SingleChanDenoiser().load()
wfs = np.swapaxes(wfs_to_denoise, 1, 2)
wfs_denoised_old = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)

# %%
wfs_denoised_old = wfs_denoised_old.detach().numpy()

# %%
from spike_psvae.denoise import SingleChanDenoiser
spk_time = 42;
dn = SingleChanDenoiser().load()

wfs_traveler = wfs_to_denoise
# maxchan = max_chan

for i in range(len(maxchan)):
    mcs = int(maxchan[i])
    ci = channel_index[mcs]
    non_nan_idx = np.where(ci<384)[0]
    
    ci = ci[non_nan_idx]
    
    
    real_maxCH = np.argmax(wfs_denoised_old[i,non_nan_idx,:].ptp(1))
    mcs_idx = real_maxCH
    # mcs_idx = np.squeeze(np.where(ci == mcs))
    previous_ch_idx = mcs_idx
    
    l = len(ci)
    
    spk_denoised_wfs = np.zeros((121, l))
    
    full_wfs = wfs_traveler[i, :, non_nan_idx].T
    
    mcs_wfs = full_wfs[:,mcs_idx]
    
    wfs = np.swapaxes(mcs_wfs[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = np.squeeze(wfs_denoised.detach().numpy())
    spk_denoised_wfs[:,mcs_idx] = wfs_denoised
    
    mcs_phase_shift = np.argmax(np.abs(wfs_denoised)) - 42
    
    
    spk_sign = np.sign(wfs_denoised[42 + mcs_phase_shift])
    
    threshold = 0.2 * wfs_denoised[42 + mcs_phase_shift]  #threshold on the peak amplitude, no phase shift if the amplitude is smaller than 20% of the maxchan
    
    

    
    # BFS to shift the phase
    ci_graph = dict()
    ci_geom = geom[ci]
    for ch in range(l):
        ci_graph[ch] = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == y_pitch))|
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * y_pitch)) |
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0)))  
    
    
    CH_checked = np.zeros(l)
    CH_phase_shift = np.zeros(l)
    parent = np.zeros(l) * np.nan
    
    parent_peak_phase = np.zeros(l)
    CH_phase_shift[mcs_idx] = mcs_phase_shift
    
    wfs_ptp = np.zeros(l)
    wfs_ptp[mcs_idx] = np.ptp(wfs_denoised)
    CH_checked[mcs_idx] = 1
    q = []
    q.append(int(mcs_idx))
    
    while len(q)>0:
        u = q.pop()
        v = ci_graph[u][0]

        for k in v:
            if CH_checked[k] == 0:
                neighbors = ci_graph[k][0]
                checked_neighbors = neighbors[CH_checked[neighbors] == 1]
                # if len(np.where(ci_geom[checked_neighbors, 0] == ci_geom[k, 0])[0])>0:
                #     phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors[ci_geom[checked_neighbors, 0] == ci_geom[k, 0]]])
                # else:
                phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors])
                    
                threshold = max(0.3* wfs_ptp[mcs_idx], 3)
                if wfs_ptp[checked_neighbors[phase_shift_ref]] > threshold:
                    parent_peak_phase[k] = CH_phase_shift[checked_neighbors[phase_shift_ref]]
                else:
                    parent_peak_phase[k] = 0

                
                CH_phase_shift[k] = denoise_with_phase_shift(full_wfs[:,k], int(parent_peak_phase[k]), k, spk_sign)
                parent[k] = checked_neighbors[phase_shift_ref]
                wfs_ptp[k] = np.ptp(spk_denoised_wfs[:,k])
                q.insert(0,k)
                CH_checked[k] = 1

    
    bias = np.arange(len(non_nan_idx))
    bias = np.repeat(bias[None,:], 121, axis = 0)
    fig, axs = plt.subplots(1 ,5 , figsize = (15, 8))

    axs[0].plot(full_wfs + bias*6, 'k')
    axs[0].plot(wfs_denoised_old[i,non_nan_idx,:].T + bias*6, 'r')
    axs[0].set_title('denoiser without phase shift')
    axs[0].vlines(42, -2, 240*l/40)
    axs[0].set_ylim(-5, 240*l/40)

    axs[1].plot(full_wfs + bias*6, 'k')
    
    # denoised_im_wfs = np.zeros((121, 40)) * np.nan
#     for j in range(40):
#         ph = np.abs(parent_peak_phase[j])
#         if parent_peak_phase[j] >= 0:
#             denoised_im_wfs[int(parent_peak_phase[j]):None,j] = spk_denoised_wfs[0:int(121 - parent_peak_phase[j]), j]
#         else:
#             denoised_im_wfs[0:int(121 - ph),j] = spk_denoised_wfs[int(ph):None, j]
            
    axs[1].plot(spk_denoised_wfs+ bias*6, 'r')
    axs[1].plot((spk_denoised_wfs+ bias*6)[:,mcs_idx], 'g')
    axs[1].vlines(42, -2, 240)
    axs[1].set_ylim(-5, 240*l/40)
    axs[1].set_title('denoiser with phase shift')

    axs[2].imshow(full_wfs[:,non_nan_idx].T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    which  = (parent_peak_phase!=0)
    axs[2].scatter(parent_peak_phase[which] + 42 , np.arange(l)[which] , c ='g')
    
    axs[2].vlines(42, 0, l)
    axs[2].set_ylim(-0.5, l - 0.5)
    axs[2].set_title('raw waveforms')
    
    
    axs[3].imshow(wfs_denoised_old[i,non_nan_idx,:], aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    # which  = (parent_peak_phase!=0)
    # axs[3].scatter(CH_phase_shift + 42, np.arange(40) , c ='g')
    axs[3].set_title('without phase-shift')
    
    
    axs[4].imshow(spk_denoised_wfs[:].T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    # which  = (parent_peak_phase!=0)
    # axs[3].scatter(CH_phase_shift + 42, np.arange(40) , c ='g')
    axs[4].set_title('phase-shift')
                
    
    
    # np.savez('/moto/stats/users/hy2562/projects/ephys_atlas/two channel denoiser/iterative_phase_shift_single_channel_denoised_unit_' + str(i) + '.npz', parent_peak_phase = parent_peak_phase, denoised_im_wfs = denoised_im_wfs)
    plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/phase-shift_single_channel_denoiser_large_scale_test/align_max_abs_ptp3/'+ 'unit_' + str(i) + 'iterative_phase_shift.png')
    plt.close()

# %%
wfs_traveler = wfs_to_denoise
# maxchan = max_chan
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/denoise_hallucination_fix/with_phase-shift'
# corrected_wfs = []
# supress_ch = []
for i in range(2569,2570):#range(len(maxchan)):
    mcs = int(maxchan[i])
    ci = channel_index[mcs]
    non_nan_idx = np.where(ci<384)[0]
    
    ci = ci[non_nan_idx]
    
    
    real_maxCH = np.argmax(wfs_denoised_old[i,non_nan_idx,:].ptp(1))
    mcs_idx = real_maxCH
    # mcs_idx = np.squeeze(np.where(ci == mcs))
    previous_ch_idx = mcs_idx
    
    l = len(ci)
    
    spk_denoised_wfs = np.zeros((121, l))
    
    full_wfs = wfs_traveler[i, :, non_nan_idx].T
    
    mcs_wfs = full_wfs[:,mcs_idx]
    
    wfs = np.swapaxes(mcs_wfs[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = np.squeeze(wfs_denoised.detach().numpy())
    spk_denoised_wfs[:,mcs_idx] = wfs_denoised
    
    mcs_phase_shift = np.argmax(np.abs(wfs_denoised)) - 42
    
    
    spk_sign = np.sign(wfs_denoised[42 + mcs_phase_shift])
    
    threshold = 0.2 * wfs_denoised[42 + mcs_phase_shift]  #threshold on the peak amplitude, no phase shift if the amplitude is smaller than 20% of the maxchan
    
    

    
    # BFS to shift the phase
    ci_graph = dict()
    ci_geom = geom[ci]
    for ch in range(l):
        ci_graph[ch] = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == y_pitch))|
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * y_pitch)) |
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0)))  
    
    
    CH_checked = np.zeros(l)
    CH_phase_shift = np.zeros(l)
    parent = np.zeros(l) * np.nan
    
    parent_peak_phase = np.zeros(l)
    CH_phase_shift[mcs_idx] = mcs_phase_shift
    
    wfs_ptp = np.zeros(l)
    wfs_ptp[mcs_idx] = np.ptp(wfs_denoised)
    CH_checked[mcs_idx] = 1
    q = []
    q.append(int(mcs_idx))
    # SC = []
    while len(q)>0:
        u = q.pop()
        v = ci_graph[u][0]

        for k in v:
            if CH_checked[k] == 0:
                neighbors = ci_graph[k][0]
                checked_neighbors = neighbors[CH_checked[neighbors] == 1]
                # if len(np.where(ci_geom[checked_neighbors, 0] == ci_geom[k, 0])[0])>0:
                #     phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors[ci_geom[checked_neighbors, 0] == ci_geom[k, 0]]])
                # else:
                phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors])
                    
                
                if wfs_ptp[checked_neighbors[phase_shift_ref]] > (0.3* wfs_ptp[mcs_idx]):
                    parent_peak_phase[k] = CH_phase_shift[checked_neighbors[phase_shift_ref]]
                else:
                    parent_peak_phase[k] = 0

                
                CH_phase_shift[k] = denoise_with_phase_shift(full_wfs[:,k], int(parent_peak_phase[k]), k, spk_sign)
                parent[k] = checked_neighbors[phase_shift_ref]
                wfs_ptp[k] = np.ptp(spk_denoised_wfs[:,k])
                q.insert(0,k)
                CH_checked[k] = 1
    
    # corrected_wfs.append(spk_denoised_wfs)
    # supress_ch.append(SC)
    
    bias = np.arange(len(non_nan_idx))
    bias = np.repeat(bias[None,:], 121, axis = 0)
    
    

    fig, axs = plt.subplots(1,4, figsize = [20, 10])
    
    axs[0].plot(full_wfs + bias*6, 'k');
    axs[0].plot(spk_denoised_wfs + bias*6, 'r');
    axs[0].set_title('single channel w shift')
    
    axs[1].plot(full_wfs+ bias*6, 'k');
    axs[1].plot(corrected_wfs[i]+ bias*6, 'r');
    sc = supress_ch[i]
    axs[1].plot((corrected_wfs[i] + bias*6)[:,sc], 'g');
    axs[1].set_title('single channel w shift + supression')
    
    axs[2].imshow(spk_denoised_wfs.T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[2].set_title('single channel')
    axs[3].imshow((corrected_wfs[i]).T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[3].set_title('single channel w supression')
    # axs[1].set_title('divide by min(c, 1)')
    # axs[2].plot(wfs_denoised_old[i,:,:].T/np.maximum(C, 1) + bias*2, 'k');
    # axs[2].set_title('divide by max(c, 1)')
    
    plt.savefig(save_dir + '/phase_shift_unit_' + str(i) + '_suppress.png')

# %%
from celluloid import Camera


# %%
walk_array = [16, 18, 19, 22, 24, 12, 14, 15, 17, 21, 23, 26, 28,  8, 10, 11, 13,
       25, 27, 30, 32,  4,  6,  7,  9, 29, 31, 34, 36,  0,  2,  3,  5, 33,
       35, 38, 39,  1, 37]
fig,axs = plt.subplots(1,2, figsize = [8, 10], gridspec_kw=dict(width_ratios=[1, 3]))
camera = Camera(fig)

for k in range(len(walk_array)):
    for i in range(len(ci_graph)):
        neighbors = ci_graph[i][0]
        for j in range(len(neighbors)):
            axs[0].plot(np.array([ci_geom[neighbors[j],0], ci_geom[i,0]]), np.array([ci_geom[neighbors[j],1], ci_geom[i,1]]), c = [0.5, 0.5, 0.5], zorder = -1)
    # plt.scatter(ci_geom[:,0], ci_geom[:,1])
    axs[0].scatter(ci_geom[20,0], ci_geom[20,1],c = 'r')

    axs[0].scatter(ci_geom[walk_array[0:(k+1)],0], ci_geom[walk_array[0:(k+1)],1],c = 'g')
    axs[0].scatter(ci_geom[walk_array[k],0], ci_geom[walk_array[k],1],c = 'pink')
    axs[0].scatter(ci_geom[np.int32(parent[walk_array[k]]),0], ci_geom[np.int32(parent[walk_array[k]]),1],c = 'b')
    
    axs[1].plot(full_wfs + bias*6, 'k')
    axs[1].plot((spk_denoised_wfs+ bias*6)[:,mcs_idx], 'r')
    axs[1].plot((spk_denoised_wfs+ bias*6)[:,walk_array[0:k]], 'g')
    axs[1].plot((spk_denoised_wfs+ bias*6)[:,walk_array[k]], 'pink')
    axs[1].plot((spk_denoised_wfs+ bias*6)[:,np.int32(parent[walk_array[k]])], 'b')
    
    axs[1].vlines(42, -2, 240)
    axs[1].set_ylim(-5, 240*l/40)
    axs[1].set_title('denoiser with phase shift')
    
    camera.snap()


animation = camera.animate()
animation.save(save_dir + '/new_BFS_vid.gif')#+ '_denoised.gif')

# %%
j = 18
s = -1
wfs = np.swapaxes(wfs_traveler[0, :, j][np.newaxis, :, np.newaxis], 1, 2)
wfs_denoised_old = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)

wfs = np.swapaxes(np.roll(wfs_traveler[0, :, j], s)[np.newaxis, :, np.newaxis], 1, 2)
wfs_denoised_new = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)

plt.plot(wfs_traveler[0, :, j])
plt.plot(np.squeeze(wfs_denoised_old.detach().numpy()))

plt.plot(np.roll(wfs_traveler[0, :, j], s))
plt.plot(np.squeeze(wfs_denoised_new.detach().numpy()))

# %%
