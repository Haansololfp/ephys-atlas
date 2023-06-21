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

# %%
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# %%
check_PIDs = dict()
check_PIDs['9117969a-3f0d-478b-ad75-98263e3bfacf'] = 3001000
check_PIDs['80f6ffdd-f692-450f-ab19-cd6d45bfd73e'] = 601000
check_PIDs['3eb6e6e0-8a57-49d6-b7c9-f39d5834e682'] = 4801000
check_PIDs['ad714133-1e03-4d3a-8427-33fc483daf1a'] = 3601000

# %%
maxchan = []

# %%
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/visualize_feature_value_on_raw'
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'
wfs_traveler = [];
maxchan = []
for i in range(4):
    pID = list(check_PIDs.keys())[i]
    eID, probe = one.pid2eid(pID)
    
    sample_start = check_PIDs[pID]
    
    spike_pick_mat = save_dir + '/pID' + pID + '_Ts_' + str(sample_start) + '_spike_pick.mat'
    spike_pick = scipy.io.loadmat(spike_pick_mat)
    
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        spike_index = h5["spike_index"][:]
        channel_index = h5["channel_index"][:]
        geom = h5["geom"][:]
    spk_times = spike_index[:,0]
    spk_channels = spike_index[:,1]
    
    destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    load_start = spk_times - 42# - 60
    load_end = spk_times + 79# + 60
    
    spk_idx = []
    for j in range(len(spike_pick['curve_travel_spike'][0])):
        spk_idx.append(np.where(spike_index[:,0]==spike_pick['curve_travel_spike'][0][j][1][0][0] + sample_start)[0][0])
        
    spk_idx = np.squeeze(np.array(spk_idx))
    
    for j in range(len(spk_idx)):
        idx = spk_idx[j]
        ci = channel_index[spk_channels[idx]]
        raw_wfs = rec.get_traces(start_frame=load_start[idx], end_frame=load_end[idx])[:,ci]
        
        wfs_traveler.append(raw_wfs)
        maxchan.append(spk_channels[idx])
wfs_traveler = np.array(wfs_traveler)

# %%
np.shape(maxchan)

# %%
np.shape(wfs_traveler)

# %%
np.save('/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/wfs_traveler.npy', wfs_traveler)

# %%
np.save('/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/wfs_traveler_maxCH.npy', maxchan)

# %%
dn = SingleChanDenoiser().load()
wfs = np.swapaxes(wfs_traveler[:,60:(60+121),:], 1, 2)
wfs_denoised_old = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)

# %%
wfs_denoised_old = wfs_denoised_old.detach().numpy()

# %% jupyter={"outputs_hidden": true}
fig, axs = plt.subplots(6, 3, figsize = (40, 100))

for i in range(17):
    row = i // 3
    col = np.mod(i, 3)
    
    bias = np.arange(40)
    bias = np.repeat(bias[None,:], 121, axis = 0)
    
    axs[row, col].plot(wfs_traveler[i,:,:] + bias*8, 'k')
    axs[row, col].plot(wfs_denoised[i,:,:].T + bias*8, 'r')

# %% [markdown]
# denoise with peak shift

# %%
from scipy.signal import argrelmin

# %% jupyter={"outputs_hidden": true}
for i in range(17):
    peaks_idx = argrelmin(wfs_traveler[i,30:64,:], axis = 0, order=40)

    threshold = 0.2 * np.min(wfs_traveler[i,:,:])

    energy = wfs_traveler[i, peaks_idx[0]+30, peaks_idx[1]]
    which = energy < threshold
    # which = slice(None)

    peaks_idx = np.array(peaks_idx)
    bias = np.arange(40)
    bias = np.repeat(bias[None,:], 121, axis = 0)
    fig, axs = plt.subplots(1 ,2 , figsize = (6, 8))
    axs[0].plot(wfs_traveler[i,:,:] + bias*13, 'k')
    axs[0].scatter(peaks_idx[0][which]+30, peaks_idx[1][which]*13 + wfs_traveler[i,peaks_idx[0]+30,peaks_idx[1]][which])

    axs[0].vlines([30, 42+22], -20, 520, 'r')


    axs[1].imshow(wfs_traveler[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[1].scatter(peaks_idx[0][which]+30, peaks_idx[1][which], c ='g')
    
    axs[1].imshow(wfs_traveler[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[1].scatter(peaks_idx[0][which]+30, peaks_idx[1][which], c ='g')

# %% [markdown]
# the phase shift can't be more than d_neighbor/v_min*fs

# %%
vel_min = 0.5*10**(-6)
fs = 30000

# %%
12 - spk_times

# %% jupyter={"outputs_hidden": true}
spk_time = 42 + 60;

for i in range(17):
    
    peaks_idx = argrelmin(wfs_traveler[i,(spk_time - 12):spk_time + 22,:], axis = 0, order=40)

    threshold = 0.2 * np.min(wfs_traveler[i,(spk_time - 12):spk_time + 22,:])

    energy = wfs_traveler[i, peaks_idx[0]+30+60, peaks_idx[1]]
    which = energy < threshold
    # which = slice(None)

    peaks_idx = np.array(peaks_idx)
    
    
    centered_range = np.arange(60,181)
    wfs = wfs_traveler[i,centered_range,:]
    
    spk_times = peaks_idx[0][which]
    spk_ch = peaks_idx[1][which]
    
    phase_shift = np.zeros(40)
    
    for j in range(len(spk_times)):
        wfs[:, spk_ch[j]] = wfs_traveler[i,centered_range - 12 + spk_times[j], spk_ch[j]]
        phase_shift[spk_ch[j]] = 12 - spk_times[j]
    
    dn = SingleChanDenoiser().load()
    wfs = np.swapaxes(wfs[None, :, :], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = wfs_denoised.detach().numpy()
    
    bias = np.arange(40)
    bias = np.repeat(bias[None,:], 121, axis = 0)
    fig, axs = plt.subplots(1 ,3 , figsize = (10, 8))

    axs[0].plot(wfs_traveler[i,centered_range,:] + bias*13, 'k')
    axs[0].plot(wfs_denoised_old[i,:,:].T + bias*13, 'r')
    axs[0].set_title('denoiser without phase shift')

    axs[1].plot(wfs_traveler[i,centered_range,:] + bias*13, 'k')
    for j in range(40):
        axs[1].plot(np.arange(121) - phase_shift[j], np.squeeze(wfs_denoised[:, j] + bias[:,j]*13), 'r')
    axs[1].set_title('denoiser with phase shift')

    axs[2].imshow(wfs_traveler[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[2].scatter(peaks_idx[0][which]+90, peaks_idx[1][which], c ='g')
    axs[2].set_title('phase shifted')
    
    plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/shift_phase_denoiser/' + 'unit_' + str(i) + 'phase_shift_from_argrelmin.png')

# %% [markdown]
# iterative phase shift

# %%
maxchan = np.array(maxchan)


# %%
def denoise_with_phase_shift(chan_long_wfs, phase_shift, chan_ci_idx, spk_sign):
    # wfs_to_denoise = np.zeros((11, 121))
    # shift = np.arange(-5, 6)
    # for s in range(11):
    #     wfs_to_denoise[s,:] = chan_long_wfs[centered_range + phase_shift + shift[s]]
    wfs_to_denoise = chan_long_wfs[centered_range + phase_shift]
    wfs = np.swapaxes(wfs_to_denoise[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = wfs_denoised.detach().numpy()
    # # print(wfs_denoised.shape())
    # # cov_ptp = np.ptp(wfs_denoised, axis = 1)
    spk_denoised_wfs[:, chan_ci_idx] = np.squeeze(wfs_denoised)
    # # spk_denoised_wfs[:, chan_ci_idx] = np.squeeze(wfs_denoised[np.argmax(cov_ptp), :])
    phase_shifted = np.argmax(wfs_denoised * spk_sign) - 42 + phase_shift #+ shift[np.argmax(cov_ptp)]
    
#     if np.ptp(wfs_denoised)>(0.2* wfs_ptp[mcs_idx]):
    
#         parent_peak_phase[k] = phase_shifted

#         wfs_to_denoise = chan_long_wfs[centered_range + phase_shifted]
#         wfs = np.swapaxes(wfs_to_denoise[None, :, None], 1, 2)
#         wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
#         wfs_denoised = wfs_denoised.detach().numpy()
#         spk_denoised_wfs[:, chan_ci_idx] = np.squeeze(wfs_denoised)
#         # phase_shifted = np.argmax(wfs_denoised * spk_sign) - 42 + phase_shifted 
#         return np.argmax(wfs_denoised * spk_sign) - 42 + phase_shifted 
    
    return phase_shifted



# %%
x_pitch = np.diff(np.unique(geom[:,0]))[0]
y_pitch = np.diff(np.unique(geom[:,1]))[0]

# %%
mcs_phase_shift

# %% jupyter={"outputs_hidden": true}
spk_time = 42 + 60;
centered_range = np.arange(60,181)
dn = SingleChanDenoiser().load()
for i in range(17):
    spk_denoised_wfs = np.zeros((121, 40))
    all_chan_phase_shift = np.zeros(40)
    mcs = maxchan[i]
    ci = channel_index[mcs]
    
    mcs_idx = np.squeeze(np.where(ci == mcs))
    previous_ch_idx = mcs_idx
    
    mcs_wfs = wfs_traveler[i, centered_range, mcs_idx]
    
    wfs = np.swapaxes(mcs_wfs[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = np.squeeze(wfs_denoised.detach().numpy())
    spk_denoised_wfs[:,mcs_idx] = wfs_denoised
    
    mcs_phase_shift = np.argmax(np.abs(wfs_denoised)) - 42
    
    
    spk_sign = np.sign(wfs_denoised[42 + mcs_phase_shift])
    
    threshold = 0.3 * wfs_denoised[42 + mcs_phase_shift]  #threshold on the peak amplitude, no phase shift if the amplitude is smaller than 20% of the maxchan
    
    
    # BFS to shift the phase
    ci_graph = dict()
    ci_geom = geom[ci]
    for ch in range(len(ci)):
        ci_graph[ch] = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == y_pitch))|
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * y_pitch)) |
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0)))  
    CH_checked = np.zeros(40)
    CH_phase_shift = np.zeros(40)
    parent = np.zeros(40) * np.nan
    
    parent_peak_phase = np.zeros(40)
    CH_phase_shift[mcs_idx] = mcs_phase_shift
    
    wfs_ptp = np.zeros(40)
    wfs_ptp[mcs_idx] = np.ptp(wfs_denoised)
    CH_checked[mcs_idx] = 1
    q = []
    q.append(int(mcs_idx))
    
    while len(q)>0:
        u = q.pop()
        v = ci_graph[u][0]
        
        CH_energy = spk_denoised_wfs[:, u]

        for k in v:
            if CH_checked[k] == 0:
                # print('ok')
                neighbors = ci_graph[k][0]
                checked_neighbors = neighbors[CH_checked[neighbors] == 1]
                phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors])
                threshold = max(0.3* wfs_ptp[mcs_idx], 3)
                if np.max(wfs_ptp[checked_neighbors]) > threshold:
                    parent_peak_phase[k] = CH_phase_shift[checked_neighbors[phase_shift_ref]]
                else:
                    parent_peak_phase[k] = 0

                
                CH_phase_shift[k] = denoise_with_phase_shift(wfs_traveler[i, :, k], int(parent_peak_phase[k]), k, spk_sign)
                parent[k] = checked_neighbors[phase_shift_ref]
                wfs_ptp[k] = np.ptp(spk_denoised_wfs[:,k])
                q.insert(0,k)
                CH_checked[k] = 1

    
    bias = np.arange(40)
    bias = np.repeat(bias[None,:], 121, axis = 0)
    fig, axs = plt.subplots(1 ,4 , figsize = (12, 8))

    axs[0].plot(wfs_traveler[i,centered_range,:] + bias*13, 'k')
    axs[0].plot(wfs_denoised_old[i,:,:].T + bias*13, 'r')
    axs[0].set_title('denoiser without phase shift')

    axs[1].plot(wfs_traveler[i,centered_range,:] + bias*13, 'k')
    
    denoised_im_wfs = np.zeros((121, 40)) * np.nan
    for j in range(40):
        ph = np.abs(parent_peak_phase[j])
        if parent_peak_phase[j] >= 0:
            denoised_im_wfs[int(parent_peak_phase[j]):None,j] = spk_denoised_wfs[0:int(121 - parent_peak_phase[j]), j]
        else:
            denoised_im_wfs[0:int(121 - ph),j] = spk_denoised_wfs[int(ph):None, j]
            
        axs[1].plot(np.arange(121) + parent_peak_phase[j], np.squeeze(spk_denoised_wfs[:, j] + bias[:,j]*13), 'r')
    axs[1].set_title('denoiser with phase shift')

    axs[2].imshow(wfs_traveler[i,centered_range,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    which  = (parent_peak_phase!=0)
    axs[2].scatter(parent_peak_phase[which] + 42 , np.arange(40)[which] , c ='g')
    axs[2].set_title('phase shifted')
    
    
    
    
    axs[3].imshow(denoised_im_wfs.T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    # which  = (parent_peak_phase!=0)
    axs[3].scatter(CH_phase_shift + 42, np.arange(40) , c ='g')
    axs[3].set_title('denoised peak')
    
    # np.savez('/moto/stats/users/hy2562/projects/ephys_atlas/two channel denoiser/iterative_phase_shift_single_channel_denoised_unit_' + str(i) + '.npz', parent_peak_phase = parent_peak_phase, denoised_im_wfs = denoised_im_wfs)
    # plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/shift_phase_denoiser/iterative_phase_shift/'+ 'unit_' + str(i) + 'iterative_phase_shift.png')

# %%
from spike_psvae.waveform_utils import closest_chans_channel_index

save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/visualize_feature_value_on_raw'
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'
wfs_traveler = [];
for i in range(4):
    pID = list(check_PIDs.keys())[i]
    eID, probe = one.pid2eid(pID)
    
    sample_start = check_PIDs[pID]
    
    spike_pick_mat = save_dir + '/pID' + pID + '_Ts_' + str(sample_start) + '_spike_pick.mat'
    spike_pick = scipy.io.loadmat(spike_pick_mat)
    
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        spike_index = h5["spike_index"][:]
        channel_index = h5["channel_index"][:]
        geom = h5["geom"][:]
        
    closest_channel_index = closest_chans_channel_index(geom, 4)
        
    spk_times = spike_index[:,0]
    spk_channels = spike_index[:,1]
    
    destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    load_start = spk_times - 42
    load_end = spk_times + 79
    
    spk_idx = []
    for j in range(len(spike_pick['curve_travel_spike'][0])):
        spk_idx.append(np.where(spike_index[:,0]==spike_pick['curve_travel_spike'][0][j][1][0][0] + sample_start)[0][0])
        
    spk_idx = np.squeeze(np.array(spk_idx))
    
    for j in range(len(spk_idx)):
        idx = spk_idx[j]
        ci = channel_index[spk_channels[idx]]
        
        closest_channel_index[spk_channels[idx]]
        
        raw_wfs = rec.get_traces(start_frame=load_start[idx], end_frame=load_end[idx])[:,ci]
        
        wfs_traveler.append(raw_wfs)
        
wfs_traveler = np.array(wfs_traveler)

# %% [markdown]
# Find collision

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/eID_b03fbc44-3d8e-4a6c-8a50-5ea3498568e0_probe_probe00_pID_9117969a-3f0d-478b-ad75-98263e3bfacf'

# %%
h5_dir = main_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    spike_index = h5["spike_index"][:]
    channel_index = h5["channel_index"][:]
    
spk_times = spike_index[:,0]
spk_channels = spike_index[:,1]



# %%
spk_dist = np.abs(spk_times[None, 0:10000].T  - spk_times[0:10000])

# %%
plt.imshow(spk_dist)

# %%
for i in range(10000):
    spk_dist[i, 0:(i - 1)] = 0

close_set = np.array(np.where((spk_dist>3) & (spk_dist<10)))

# %%
pick_idx = []
for i in range(np.shape(close_set)[1]):
    if np.abs(spk_channels[close_set[0, i]] - spk_channels[close_set[1, i]])<=2:
        pick_idx.append(close_set[:,i])

# %%
pick_idx = np.array(pick_idx)

# %%
np.argsort(spk_times[pick_idx[i,:]])

# %%
import matplotlib as mpl
fig, axs = plt.subplots(10, 10, figsize = (20, 20))

destriped_cbin_dir = list(Path(main_dir).glob('destriped_*.cbin'))[0]
destriped_cbin = main_dir + '/' + destriped_cbin_dir.name

rec_cbin = sf.read_cbin_ibl(Path(main_dir))
rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

load_start = spk_times - 42
load_end = spk_times + 79

raw_collision_example = []
for i in range(100):
    idx = pick_idx[i, 0]
    s_t = spk_times[idx]
    load_start = s_t - 42
    load_end = s_t + 79
    
    ci = channel_index[spk_channels[idx]]
    
    ci_not_nan = ci[ci<384]
    idx_ci_not_nan = np.squeeze(np.where(ci<384))
    raw_wfs = np.zeros((121, 40))*np.nan
    
    raw_wfs[:, idx_ci_not_nan] = rec.get_traces(start_frame=load_start, end_frame=load_end)[:,ci_not_nan]
    row = i // 10
    col = np.mod(i, 10)
    axs[row][col].imshow(raw_wfs.T, aspect = 'auto', vmin = -2, vmax = 2, cmap=mpl.colormaps['RdBu'], origin='lower')
    
    raw_collision_example.append(raw_wfs)

# %%
destriped_cbin_dir = list(Path(main_dir).glob('destriped_*.cbin'))[0]
destriped_cbin = main_dir + '/' + destriped_cbin_dir.name

rec_cbin = sf.read_cbin_ibl(Path(main_dir))
rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

load_start = spk_times - 42
load_end = spk_times + 79

maxchan = []
raw_collision_example = []
second_t = []
for i in range(100):
    sort_idx = np.argsort(spk_times[pick_idx[i,:]])
    pick_idx[i,:] = pick_idx[i,sort_idx]
    idx = pick_idx[i, 0]
    s_t = spk_times[idx]

    
    load_start = s_t - 42 - 60
    load_end = s_t + 79 + 60
    
    ci = channel_index[spk_channels[idx]]
    
    ci_not_nan = ci[ci<384]
    idx_ci_not_nan = np.squeeze(np.where(ci<384))
    
    if len(idx_ci_not_nan)==40:
        raw_wfs = rec.get_traces(start_frame=load_start, end_frame=load_end)[:,ci]
        raw_collision_example.append(raw_wfs)
        maxchan.append([spk_channels[idx], spk_channels[pick_idx[i, 1]]])
        second_t.append(spk_times[pick_idx[i, 1]] -  spk_times[pick_idx[i, 0]])

# %%
np.savez('/moto/stats/users/hy2562/projects/ephys_atlas/two channel denoiser/collision_examples.npz', raw_collision_example = raw_collision_example, maxchan = maxchan, second_t = second_t)

# %% jupyter={"outputs_hidden": true}
bias = bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)

for i in range(100):
    
    plt.figure(figsize = (10, 20))
    plt.plot(raw_collision_example[i] + bias *5);
    # plt.vlines([42, 42 + spk_times[pick_idx[i, 1]] -  spk_times[pick_idx[i, 0]]], -10, 200)

    mc1 = spk_channels[pick_idx[i, 0]]
    mcs_idx = np.squeeze(np.where(channel_index[mc1] == mc1))
    mc2 = spk_channels[pick_idx[i, 1]] - spk_channels[pick_idx[i, 0]] + mcs_idx
    plt.scatter(42, raw_collision_example[i][42, mcs_idx] + mcs_idx*5, c ='r')
    plt.scatter(42 + spk_times[pick_idx[i, 1]] -  spk_times[pick_idx[i, 0]], raw_collision_example[i][42 + spk_times[pick_idx[i, 1]] - spk_times[pick_idx[i, 0]], mc2] + mc2*5, c = 'r')
    
    plt.title('unit_' + str(i))


# %%
def denoise_with_phase_shift(chan_long_wfs, phase_shift, chan_ci_idx, spk_sign):
    # wfs_to_denoise = np.zeros((11, 121))
    # shift = np.arange(-5, 6)
    # for s in range(11):
    #     wfs_to_denoise[s,:] = chan_long_wfs[centered_range + phase_shift + shift[s]]
    wfs_to_denoise = chan_long_wfs[centered_range + phase_shift]
    wfs = np.swapaxes(wfs_to_denoise[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = wfs_denoised.detach().numpy()
    # # print(wfs_denoised.shape())
    # # cov_ptp = np.ptp(wfs_denoised, axis = 1)
    spk_denoised_wfs[:, chan_ci_idx] = np.squeeze(wfs_denoised)
    # # spk_denoised_wfs[:, chan_ci_idx] = np.squeeze(wfs_denoised[np.argmax(cov_ptp), :])
    phase_shifted = np.argmax(wfs_denoised * spk_sign) - 42 + phase_shift #+ shift[np.argmax(cov_ptp)]
    
#     if np.ptp(wfs_denoised)>(0.2* wfs_ptp[mcs_idx]):
    
#         parent_peak_phase[k] = phase_shifted

#         wfs_to_denoise = chan_long_wfs[centered_range + phase_shifted]
#         wfs = np.swapaxes(wfs_to_denoise[None, :, None], 1, 2)
#         wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
#         wfs_denoised = wfs_denoised.detach().numpy()
#         spk_denoised_wfs[:, chan_ci_idx] = np.squeeze(wfs_denoised)
#         # phase_shifted = np.argmax(wfs_denoised * spk_sign) - 42 + phase_shifted 
#         return np.argmax(wfs_denoised * spk_sign) - 42 + phase_shifted 
    
    return phase_shifted



maxchan = np.array(maxchan)

spk_time = 42 + 60;
dn = SingleChanDenoiser().load()

wfs_traveler = np.array(raw_collision_example)




for i in range(20):
    centered_range = np.arange(60,181)
    spk_denoised_wfs = np.zeros((121, 40))
    all_chan_phase_shift = np.zeros(40)
    mcs = maxchan[i, 0]
    ci = channel_index[mcs]
    
    mcs_idx = np.squeeze(np.where(ci == mcs))
    previous_ch_idx = mcs_idx
    
    mcs_wfs = wfs_traveler[i, centered_range, mcs_idx]
    
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
    for ch in range(len(ci)):
        ci_graph[ch] = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == y_pitch))|
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * y_pitch)) |
                           ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0)))  
    CH_checked = np.zeros(40)
    CH_phase_shift = np.zeros(40)
    parent = np.zeros(40) * np.nan
    
    parent_peak_phase = np.zeros(40)
    CH_phase_shift[mcs_idx] = mcs_phase_shift
    
    wfs_ptp = np.zeros(40)
    wfs_ptp[mcs_idx] = np.ptp(wfs_denoised)
    CH_checked[mcs_idx] = 1
    q = []
    q.append(int(mcs_idx))
    
    while len(q)>0:
        u = q.pop()
        v = ci_graph[u][0]
        
        CH_energy = spk_denoised_wfs[:, u]

        for k in v:
            if CH_checked[k] == 0:
                # print('ok')
                neighbors = ci_graph[k][0]
                checked_neighbors = neighbors[CH_checked[neighbors] == 1]
                phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors])
                
                if np.max(wfs_ptp[checked_neighbors]) > (0.2* wfs_ptp[mcs_idx]):
                    parent_peak_phase[k] = CH_phase_shift[checked_neighbors[phase_shift_ref]]
                else:
                    parent_peak_phase[k] = 0

                
                CH_phase_shift[k] = denoise_with_phase_shift(wfs_traveler[i, :, k], int(parent_peak_phase[k]), k, spk_sign)
                parent[k] = neighbors[phase_shift_ref]
                wfs_ptp[k] = np.ptp(spk_denoised_wfs[:,k])
                q.insert(0, k)
                CH_checked[k] = 1

    
    bias = np.arange(40)
    bias = np.repeat(bias[None,:], 121, axis = 0)
    fig, axs = plt.subplots(2 ,4 , figsize = (12, 16))

    unshifted_wfs1 = wfs_traveler[i, centered_range, :]
    
    wfs = np.swapaxes(unshifted_wfs1[None, :, :], 1, 2)
    wfs_denoised_unshifted = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised_unshifted = np.squeeze(wfs_denoised_unshifted.detach().numpy())

    
    axs[0][0].plot(wfs_traveler[i,centered_range,:] + bias*8, 'k')
    
    denoised_im_wfs = np.zeros((121, 40)) * np.nan
    for j in range(40):
        ph = np.abs(parent_peak_phase[j])
        if parent_peak_phase[j] >= 0:
            denoised_im_wfs[int(parent_peak_phase[j]):None,j] = spk_denoised_wfs[0:int(121 - parent_peak_phase[j]), j]
        else:
            denoised_im_wfs[0:int(121 - ph),j] = spk_denoised_wfs[int(ph):None, j]
            
        axs[0][0].plot(np.arange(121) + parent_peak_phase[j], np.squeeze(spk_denoised_wfs[:, j] + bias[:,j]*8), 'r')
    axs[0][0].set_title('first spike')
    
    axs[1][0].plot(wfs_traveler[i,centered_range,:] + bias*8, 'k')
    axs[1][0].plot(wfs_denoised_unshifted.T + bias*8, 'b')
    axs[1][0].set_title('first spike no shift')
    
    axs[0][1].imshow(denoised_im_wfs.T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    # which  = (parent_peak_phase!=0)
    axs[0][1].scatter(CH_phase_shift + 42, np.arange(40) , c ='g', s = 10)
    axs[0][1].set_title('denoised peak')
    
    axs[1][1].imshow(wfs_denoised_unshifted, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[1][1].scatter(np.argmin(wfs_denoised_unshifted, axis = 1), np.arange(40) , c ='g', s = 10)
    axs[1][1].set_title('denoised peak no shift')
    ###############################
    
    spk_denoised_wfs = np.zeros((121, 40))
    all_chan_phase_shift = np.zeros(40)
    mcs = maxchan[i, 1]
    
    mcs_idx = np.squeeze(np.where(ci == mcs))
    previous_ch_idx = mcs_idx
    
    mcs_wfs = wfs_traveler[i, centered_range + second_t[i], mcs_idx]
    
    wfs = np.swapaxes(mcs_wfs[None, :, None], 1, 2)
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = np.squeeze(wfs_denoised.detach().numpy())
    spk_denoised_wfs[:,mcs_idx] = wfs_denoised
    
    mcs_phase_shift = np.argmax(np.abs(wfs_denoised)) - 42
    
    
    spk_sign = np.sign(wfs_denoised[42 + mcs_phase_shift])
    threshold = 0.2 * wfs_denoised[42 + mcs_phase_shift]  #threshold on the peak amplitude, no phase shift if the amplitude is smaller than 20% of the maxchan
    
    centered_range = centered_range + second_t[i]
    
    # BFS to shift the phase
    
    CH_checked = np.zeros(40)
    CH_phase_shift = np.zeros(40)
    parent = np.zeros(40) * np.nan
    
    parent_peak_phase = np.zeros(40)
    CH_phase_shift[mcs_idx] = mcs_phase_shift
    
    wfs_ptp = np.zeros(40)
    wfs_ptp[mcs_idx] = np.ptp(wfs_denoised)
    CH_checked[mcs_idx] = 1
    q = []
    q.append(int(mcs_idx))
    
    while len(q)>0:
        u = q.pop()
        v = ci_graph[u][0]
        
        CH_energy = spk_denoised_wfs[:, u]

        for k in v:
            if CH_checked[k] == 0:
                # print('ok')
                neighbors = ci_graph[k][0]
                checked_neighbors = neighbors[CH_checked[neighbors] == 1]
                phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors])
                
                if np.max(wfs_ptp[checked_neighbors]) > (0.2* wfs_ptp[mcs_idx]):
                    parent_peak_phase[k] = CH_phase_shift[checked_neighbors[phase_shift_ref]]
                else:
                    parent_peak_phase[k] = 0

                
                CH_phase_shift[k] = denoise_with_phase_shift(wfs_traveler[i, :, k], int(parent_peak_phase[k]), k, spk_sign)
                parent[k] = neighbors[phase_shift_ref]
                wfs_ptp[k] = np.ptp(spk_denoised_wfs[:,k])
                q.append(k)
                CH_checked[k] = 1
    
    
    unshifted_wfs2 = wfs_traveler[i, centered_range, :]
    
    wfs = np.swapaxes(unshifted_wfs2[None, :, :], 1, 2)
    wfs_denoised_unshifted = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised_unshifted = np.squeeze(wfs_denoised_unshifted.detach().numpy())
    
    axs[0][2].plot(wfs_traveler[i,centered_range,:] + bias*8, 'k')
    
    denoised_im_wfs = np.zeros((121, 40)) * np.nan
    for j in range(40):
        ph = np.abs(parent_peak_phase[j])
        if parent_peak_phase[j] >= 0:
            denoised_im_wfs[int(parent_peak_phase[j]):None,j] = spk_denoised_wfs[0:int(121 - parent_peak_phase[j]), j]
        else:
            denoised_im_wfs[0:int(121 - ph),j] = spk_denoised_wfs[int(ph):None, j]
            
        axs[0][2].plot(np.arange(121) + parent_peak_phase[j], np.squeeze(spk_denoised_wfs[:, j] + bias[:,j]*8), 'r')
    axs[0][2].set_title('second spike')
    
    axs[1][2].plot(wfs_traveler[i,centered_range,:] + bias*8, 'k')
    axs[1][2].plot(wfs_denoised_unshifted.T + bias*8, 'b')
    axs[1][2].set_title('second spike no shift')
    
    
    axs[0][3].imshow(denoised_im_wfs.T, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[0][3].scatter(CH_phase_shift + 42, np.arange(40) , c ='g', s = 10)
    axs[0][3].set_title('denoised peak')
    
    axs[1][3].imshow(wfs_denoised_unshifted, aspect = 'auto', cmap = 'RdBu', vmin = -3, vmax = 3, origin = 'lower')
    axs[1][3].scatter(np.argmin(wfs_denoised_unshifted, axis = 1), np.arange(40) , c ='g', s = 10)
    axs[1][3].set_title('denoised peak')
    plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/shift_phase_denoiser/iterative_phase_shift_collision/'+ 'unit_' + str(i) + 'iterative_phase_shift.png')

# %%
