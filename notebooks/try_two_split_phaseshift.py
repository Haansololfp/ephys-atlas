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
from spike_psvae.waveform_utils import make_channel_index, make_contiguous_channel_index
import h5py
import numpy as np
from spike_psvae.denoise import SingleChanDenoiser
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# %%
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
h5_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_69c9a415-f7fa-4208-887b-1417c1479b48_probe_probe00_pID_1a276285-8b0e-4cc9-9f0a-a3a002978724'
h5_dir = h5_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    channel_index = h5['channel_index'][:]
    geom = h5['geom'][:]

# %%
extract_box_radius = 200
box_norm_p = 2
extract_channel_index = make_channel_index(
            geom, extract_box_radius, distance_order=False, p=box_norm_p
        )

# %%
fig, axs = plt.subplots(5, 20, figsize = [20, 20])
for i in range(100):
    CH_ci = extract_channel_index[i,:]
    row = i//20
    col = np.mod(i, 20)
    which = (CH_ci < 384)
    axs[row][col].scatter(geom[CH_ci[which], 0], geom[CH_ci[which], 1])
    axs[row][col].scatter(geom[i, 0], geom[i, 1])
    axs[row][col].get_xaxis().set_ticks([])
    axs[row][col].get_yaxis().set_ticks([])
plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/two_split_phaseshift/probe_layout.png')
# extract_channel_index

# %%
radius = 50
channel_index = make_channel_index(geom, radius, pad_val = 40)

# %%
channel_index_all = np.full((384, 40, 8), 40)
maxchan_on_mask = np.zeros(384)
for i in range(384):
    ci = extract_channel_index[i]
    ci_on_probe = ci[ci<384]
    geom_on_probe = geom[ci_on_probe,:]
    channel_index = make_channel_index(geom_on_probe, radius, pad_val = 40)
    l = len(channel_index)
    channel_index_all[i,:l,:] = channel_index
    maxchan_on_mask[i] = np.where(ci_on_probe == i)[0] 

maxchan_on_mask = np.int32(maxchan_on_mask)

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
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'
template_raw_wfs_benchmark = np.load(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy')
template_raw_wfs_benchmark = template_raw_wfs_benchmark.item()

# %%
radius = 50
channel_index = make_channel_index(geom, radius)

# %%
Denoiser = SingleChanDenoiser().load()

# %%
wfs_to_denoise = []
max_channels_to_denoise = []
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    wfs = template_raw_wfs_benchmark[pID]['wfs']
    temp = template_raw_wfs_benchmark[pID]['temp']
    max_channels = template_raw_wfs_benchmark[pID]['maxchan']
    units = list(wfs.keys())
    for j in range(len(units)):
        unit_wfs = wfs[units[j]]
        template = temp[units[j]]
        wfs_to_denoise.append(unit_wfs)
        max_channels_to_denoise.append(np.ones(100,)*max_channels[j])

# %%
wfs_to_denoise = np.array(wfs_to_denoise)
max_channels_to_denoise = np.array(max_channels_to_denoise)

# %%
wfs_to_denoise = torch.tensor(wfs_to_denoise)
max_channels_to_denoise = torch.tensor(max_channels_to_denoise)


# %%
def ptp(t, axis):
    # ptp for torch
    t = torch.nan_to_num(t, nan=0.0)
    return t.max(axis).values - t.min(axis).values


# %%
def two_split_phase_shit_denoise(waveforms, maxchans, maxchans_on_mask, channel_index, denoiser, offset = 42):
    N, T, C = waveforms.shape
    waveforms = F.pad(waveforms, (0, 1), 'constant',0)
    spike_idx_unfold = torch.unsqueeze(torch.arange(N), 1).repeat(1, channel_index.shape[2]).reshape(-1)
    # maxchan_unfold = torch.unsqueeze(maxchans, 1).repeat(1, 8).reshape(-1)
    max_neighbor_unfold = channel_index[maxchans, maxchans_on_mask[maxchans],:].reshape(-1)
    maxCH_neighbor_wfs = waveforms[spike_idx_unfold, :, max_neighbor_unfold]

    denoised_maxCH_neighbor_wfs = denoiser(maxCH_neighbor_wfs.reshape(-1, T)).reshape([N, -1, T])
    denoised_maxCH_neighbor_wfs = denoised_maxCH_neighbor_wfs.transpose(2, 1)
    
    denoised_maxCH_neighbor_ptp = ptp(denoised_maxCH_neighbor_wfs, 1)
    denoised_maxCH_neighbor_maxptp_info = torch.max(denoised_maxCH_neighbor_ptp, 1)
    
    denoised_maxCH_neighbor_maxptp = denoised_maxCH_neighbor_maxptp_info[0]
    real_maxCH = denoised_maxCH_neighbor_maxptp_info[1]
    
    spk_signs = torch.sign(denoised_maxCH_neighbor_wfs[range(N), offset, real_maxCH])
    
    # print(torch.argmax(torch.swapaxes(denoised_maxCH_neighbor_wfs, 0, 2) * spk_signs, 1) )
    # print(torch.swapaxes(denoised_maxCH_neighbor_wfs, 0, 2).shape)
    phase_shifted = (torch.argmax(torch.swapaxes(denoised_maxCH_neighbor_wfs, 0, 2) * spk_signs, 1) - offset).transpose(0,1)  # Nx8
    # phase_shifted = (torch.argmin(denoised_maxCH_neighbor_wfs, 1) - offset)  # Nx8
    # print(type(max_neighbor_unfold))
    on_probe_idx = torch.squeeze(torch.nonzero(max_neighbor_unfold<C), 1)
    
    
    top_border = torch.stack(list(torch.max(max_neighbor_unfold[on_probe_idx][spike_idx_unfold[on_probe_idx] == i]) for i in range(N)))
    low_border = torch.stack(list(torch.min(max_neighbor_unfold[on_probe_idx][spike_idx_unfold[on_probe_idx] == i]) for i in range(N)))
    
    
    thresholds = torch.max(0.3*denoised_maxCH_neighbor_maxptp, torch.tensor(3))
    threshold_accept_idx = (denoised_maxCH_neighbor_ptp.reshape(-1) > thresholds[spike_idx_unfold]).reshape(N, -1)
    
    
    
    phase_shifted_diff = torch.unsqueeze(phase_shifted, 2) - torch.unsqueeze(phase_shifted, 1) 
    triu_index = torch.triu_indices(channel_index.shape[2], channel_index.shape[2], -1)
    phase_shifted_diff[:, triu_index[0,:], triu_index[1,:]] = 100
    phase_accept_idx = (torch.min(phase_shifted_diff, 2)[0] <=5)
    
    # print(threshold_accept_idx.shape)
    # print(phase_accept_idx.shape)
    # print(phase_shifted.shape)
    phase_shifted = torch.where(threshold_accept_idx & phase_accept_idx, phase_shifted, 100)
    
    c = torch.nonzero(phase_shifted <100)
    
    # print(phase_shifted)
    top_phase = torch.stack(list(torch.max(c[:,1][c[:,0] == i]) for i in range(N)))
    low_phase = torch.stack(list(torch.min(c[:,1][c[:,0] == i]) for i in range(N)))
    
    top_border_phase_shift = phase_shifted[range(N), top_phase]
    low_border_phase_shift = phase_shifted[range(N), low_phase]
    
    # print(top_border_phase_shift)
    phase_shift_pick_array = torch.zeros(N, C)
    
    for i in range(N):
        if top_border[i] < C-1:
            phase_shift_pick_array[i, top_border[i]+1:] = top_border_phase_shift[i]
        if low_border[i]>0:
            phase_shift_pick_array[i, :low_border[i]-1] = low_border_phase_shift[i]
            
    phase_shift_pick_array = phase_shift_pick_array.reshape(-1).long()
    wfs = waveforms[:, :, :C].transpose(2, 1).reshape(-1, T)
    
    
    rolled_wfs = roll_by_gather(wfs, -phase_shift_pick_array)
    
    denoised_wfs = denoiser(rolled_wfs)
    
    denoised_wfs = roll_by_gather(denoised_wfs, phase_shift_pick_array).reshape(N, -1, T)
    
    return denoised_wfs.transpose(2, 1), phase_shift_pick_array.reshape(N, -1)


# %%
def roll_by_gather(wfs, shifts: torch.LongTensor):
    # assumes 2D array
    N, T = wfs.shape
    arange1 = torch.arange(T).view((1,T)).repeat((N,1))
    # print(arange1.shape)
    arange2 = (arange1 - shifts[:, None]) % T
    return torch.gather(wfs, 1, arange2)


# %%
wfs_traveler = np.load('/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/wfs_traveler.npy')
traveler_maxchans = np.load('/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/wfs_traveler_maxCH.npy')

# %%
wfs_traveler = torch.tensor(wfs_traveler)
traveler_maxchans = torch.tensor(traveler_maxchans)

# %%
denoised_wfs, phase_shift = two_split_phase_shit_denoise(wfs_traveler,traveler_maxchans, torch.tensor(maxchan_on_mask), torch.tensor(channel_index_all), Denoiser)

# %%
bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)

# %%
plt.plot(phase_shift[12,:]);

# %%
denoised_wfs = denoised_wfs.detach().numpy()
wfs_traveler = wfs_traveler.detach().numpy()


# %%
wfs_traveler = torch.tensor(wfs_traveler)
N, T, C = wfs_traveler.shape
denoised_wfs_old = Denoiser(wfs_traveler.swapaxes(1,2).reshape(-1, T)).reshape(N, C, T)
denoised_wfs_old = denoised_wfs_old.detach().numpy()

# %%
denoised_wfs_old = denoised_wfs_old.swapaxes(1,2)

# %% jupyter={"outputs_hidden": true}
bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
for i in range(17):
    fig, axs = plt.subplots(1, 5, figsize = (15, 8))
    axs[0].plot(wfs_traveler[i,:,:] + bias*10, 'k')
    axs[0].plot(denoised_wfs_old[i,:,:] + bias*10, 'r')
    axs[0].set_title('old denoiser')
    
    axs[1].plot(wfs_traveler[i,:,:] + bias*10, 'k')
    axs[1].plot(denoised_wfs[i,:,:] + bias*10, 'g')
    axs[1].set_title('phase-shift denoiser')
    
    axs[2].imshow(wfs_traveler[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[2].set_title('raw wavefors')
    
    axs[3].imshow(denoised_wfs_old[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[3].set_title('old denoiser')
    
    axs[4].imshow(denoised_wfs[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[4].set_title('phase-shift denoiser')
    
    plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/two_split_phaseshift/traveling_spikes_check'+ '/traveling_unit' + str(i) + '_two_spit_phaseshift_denoise.png')
    # plt.plot(denoised_wfs_old[i,:,:40,0] + bias*10, 'k');
    # plt.plot(denoised_wfs_array[i,:,:] + bias*10, 'r');

# %%
radius = 200
channel_index = make_channel_index(geom, radius)


# %%
def probe_segment(channel_index, C, channel_n):
    '''
    channel_n is the number of channels we grab for each spike
    '''
    N = geom.shape[0]
    maxCH_neighbor = np.full((N, 10), C)
    top_probe_blocks = np.full((N, 4, 4), C)
    low_probe_blocks = np.full((N, 4, 4), C)
    
    for i in range(N):
        idx = np.where(channel_index[i] == i)[0]
        pick_idx = np.arange(-4, 6) + idx - np.mod(i, 2)
            
        which = np.where((pick_idx>=0) & (pick_idx<C))[0]
        maxCH_on_probe = pick_idx[which]
        
        maxCH_neighbor[i, :len(which)] = maxCH_on_probe
        
        top_border = np.max(maxCH_neighbor[i,:])
        low_border = np.min(maxCH_neighbor[i,:])
        
        for j in range(4):
            top_block = np.arange(4) + top_border + 1
            which = np.where((top_block>=0) & (top_block<C))[0]
            top_block = top_block[which]
            
            which = np.where(channel_index[i, top_block]<channel_n)[0]
            top_block = top_block[which]
            
            top_probe_blocks[i, j, :len(top_block)] = top_block
            top_border = np.max(top_probe_blocks[i, j, :])
            
            low_block = - np.arange(4) + low_border - 1
            which = np.where((low_block>=0) & (low_block<C))[0]
            low_block = np.sort(low_block[which])
            
            which = np.where(channel_index[i, low_block]<channel_n)[0]
            low_block = low_block[which]

            low_probe_blocks[i, j, :len(low_block)] = low_block
            low_border = np.min(low_probe_blocks[i, j, :])
            
    return maxCH_neighbor, top_probe_blocks, low_probe_blocks


# %%
maxCH_neighbor, top_probe_blocks, low_probe_blocks = probe_segment(channel_index, 40, 384)

# %%
i = 28
ci = channel_index[i]
idx = maxCH_neighbor[i]

maxCH_loc = np.where(ci == i)[0]
on_probe_idx = np.where(idx<40)[0]
plt.figure(figsize = [3, 10])
plt.scatter(geom[ci[idx[on_probe_idx]],0], geom[ci[idx[on_probe_idx]],1])
for j in range(4):
    idx = top_probe_blocks[i, j, :]
    on_probe_idx = np.where(idx<40)[0]
    plt.scatter(geom[ci[idx[on_probe_idx]],0], geom[ci[idx[on_probe_idx]],1])
for j in range(4):
    idx = low_probe_blocks[i, j, :]
    on_probe_idx = np.where(idx<40)[0]
    plt.scatter(geom[ci[idx[on_probe_idx]],0], geom[ci[idx[on_probe_idx]],1])
    
plt.scatter(geom[ci[maxCH_loc],0], geom[ci[maxCH_loc],1],  c = 'k')


# %%
def wfs_corr(wfs_raw, wfs_denoise):
    return torch.sum(wfs_denoise*wfs_raw, 1)/torch.sqrt(torch.sum(wfs_raw*wfs_raw,1) * torch.sum(wfs_denoise*wfs_denoise,1))


# %%
def hallucination_idx_compute(chan_wfs, wfs_denoised, offset = 42, small_threshold = 2, corr_th = 0.8):
    which = slice(offset-10, offset+10)
    d_s_corr = wfs_corr(chan_wfs[:, which], wfs_denoised[:, which])#torch.sum(wfs_denoised[which]*chan_wfs[which], 1)/torch.sqrt(torch.sum(chan_wfs[which]*chan_wfs[which],1) * torch.sum(wfs_denoised[which]*wfs_denoised[which],1)) ## didn't use which at the beginning! check whether this changes the results
    halu_idx = (ptp(wfs_denoised, 1)<small_threshold) & (d_s_corr<corr_th)
    return halu_idx


# %%
def block_segment_phase_shit_denoise(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, denoiser, offset = 42):
    N, T, C = waveforms.shape
    waveforms = F.pad(waveforms, (0, 1), 'constant',0)
    denoised_wfs = torch.full(waveforms.shape, 0).float()
    # phase_shift = torch.zeros((N, C+1)).long()
    
    
    spike_idx_unfold = torch.unsqueeze(torch.arange(N), 1).repeat(1, maxCH_neighbor.shape[1]).reshape(-1)
    
    
    max_neighbor_unfold = maxCH_neighbor[maxchans, :].reshape(-1)
    maxCH_neighbor_wfs = waveforms[spike_idx_unfold, :, max_neighbor_unfold]
    

    denoised_maxCH_neighbor_wfs = denoiser(maxCH_neighbor_wfs)
    
    denoised_wfs[spike_idx_unfold, :, max_neighbor_unfold] = denoised_maxCH_neighbor_wfs
    
    denoised_maxCH_neighbor_wfs = denoised_maxCH_neighbor_wfs.reshape([N, -1, T]).transpose(2, 1)
    

    denoised_maxCH_neighbor_ptp = ptp(denoised_maxCH_neighbor_wfs, 1)
    denoised_maxCH_neighbor_maxptp_info = torch.max(denoised_maxCH_neighbor_ptp, 1)
    
    denoised_maxCH_neighbor_maxptp = denoised_maxCH_neighbor_maxptp_info[0]
    real_maxCH = denoised_maxCH_neighbor_maxptp_info[1]
    
    spk_signs = torch.sign(denoised_maxCH_neighbor_wfs[range(N), offset, real_maxCH])
    
   

    phase_shifted = (torch.argmax(torch.swapaxes(denoised_maxCH_neighbor_wfs, 0, 2) * spk_signs, 1) - offset).transpose(0,1)  # Nx8
    

    
    thresholds = torch.max(0.3*denoised_maxCH_neighbor_maxptp, torch.tensor(3))
    threshold_accept_idx = (denoised_maxCH_neighbor_ptp.reshape(-1) > thresholds[spike_idx_unfold]).reshape(N, -1)
    
   

    phase_shifted_diff = torch.unsqueeze(phase_shifted, 2) - torch.unsqueeze(phase_shifted, 1) 
    triu_index = torch.triu_indices(maxCH_neighbor.shape[1], maxCH_neighbor.shape[1], 0)
    phase_shifted_diff[:, triu_index[0,:], triu_index[1,:]] = 100
    phase_accept_idx = (torch.min(phase_shifted_diff, 2)[0] <=5)
    
    

    phase_shifted = torch.where(threshold_accept_idx & phase_accept_idx, phase_shifted, T)
    
    c = (phase_shifted < T)
    
    idx = torch.arange(0, c.shape[1])
    c2 = c * idx
    # pick the phase-shifts on borders as the reference phase-shifts
    top_phase = torch.argmax(c2, 1)
    
    idx = torch.arange(c.shape[1], 0, -1)
    c2 = c * idx
    
    low_phase = torch.argmin(c2, 1)
    # may be problematic, check later!!

    

    top_border_phase_shift = phase_shifted[range(N), top_phase]
    low_border_phase_shift = phase_shifted[range(N), low_phase]
    
    block_C = top_probe_blocks.shape[2]
    phase_shift_pick_array = torch.cat((top_border_phase_shift[:, None].repeat(1, block_C), low_border_phase_shift[:, None].repeat(1, block_C)), 0).reshape(-1)
    
    
    hallu_idx = (torch.sum(hallucination_idx_compute(maxCH_neighbor_wfs, denoised_wfs[spike_idx_unfold, :, max_neighbor_unfold]).reshape(N,-1), 1)>3)
    
    
    top_suppress_idx = hallu_idx.clone()
    low_suppress_idx = hallu_idx.clone()
    
    
    for i in range(top_probe_blocks.shape[1]):
        top_block = top_probe_blocks[maxchans, i, :]
        low_block = low_probe_blocks[maxchans, i, :]
        
        # print(low_block)
        
        top_block[top_suppress_idx == 1,:] = C
        low_block[low_suppress_idx == 1,:] = C
        
        spike_idx_unfold = torch.unsqueeze(torch.arange(N), 1).repeat(1, top_probe_blocks.shape[2]).reshape(-1)
        
        
        top_block_unfold = top_block.reshape(-1)
        low_block_unfold = low_block.reshape(-1)
        
        
        top_block_wfs = waveforms[spike_idx_unfold, :, top_block_unfold]
        
        
        low_block_wfs = waveforms[spike_idx_unfold, :, low_block_unfold]
        
        wfs_to_denoise = torch.cat((top_block_wfs, low_block_wfs), 0)

        wfs = wfs_to_denoise
        
        # print(torch.cat((spike_idx_unfold[:,None].repeat(2, 1),phase_shift_pick_array[:, None], torch.cat((top_block_unfold[:, None], low_block_unfold[:, None]), 0)), 1))
        # print(phase_shift_pick_array.shape)
        ########
        # phase_shift[torch.squeeze(spike_idx_unfold[:,None].repeat(2, 1)), torch.cat((top_block_unfold[:, None], low_block_unfold[:, None]), 0)] = phase_shift_pick_array
        
        rolled_wfs = roll_by_gather(wfs, -phase_shift_pick_array)
    
        denoised_top_low_block_wfs = denoiser(rolled_wfs)
        
        
        # hallu_idx = hallucination_idx_compute(rolled_wfs, denoised_top_low_block_wfs)
        
        top_hallu_idx = (torch.sum(hallucination_idx_compute(rolled_wfs[:N*block_C, :], denoised_top_low_block_wfs[:N*block_C, :]).reshape(N,-1), 1)>3)
        low_hallu_idx = (torch.sum(hallucination_idx_compute(rolled_wfs[N*block_C:, :], denoised_top_low_block_wfs[N*block_C:, :]).reshape(N,-1), 1)>3)
        
        # print(low_suppress_idx)
        # print(low_hallu_idx)
        
#         print(top_suppress_idx)
#         print(low_suppress_idx)
        
#         print(torch.nonzero(top_hallu_idx))
#         print(torch.nonzero(low_hallu_idx))
        
        top_suppress_idx[torch.squeeze(torch.nonzero(top_hallu_idx))] = 1
        low_suppress_idx[torch.squeeze(torch.nonzero(low_hallu_idx))] = 1
       
        
        denoised_top_low_block_wfs = roll_by_gather(denoised_top_low_block_wfs, phase_shift_pick_array).reshape(2*N, -1, T)
        
        
        denoised_wfs[spike_idx_unfold, :, top_block_unfold] = denoised_top_low_block_wfs[:N, :, :].reshape(-1, T)
        denoised_wfs[spike_idx_unfold, :, low_block_unfold] = denoised_top_low_block_wfs[N:, :, :].reshape(-1, T)

        
        
        denoised_top_low_block_wfs = denoised_top_low_block_wfs.transpose(2, 1)

        denoised_top_low_block_ptp = ptp(denoised_top_low_block_wfs, 1)
        
        denoised_top_block_ptp = denoised_top_low_block_ptp[:N]
        denoised_low_block_ptp = denoised_top_low_block_ptp[N:]
        
        
        
        
        top_threshold_accept_idx = (denoised_top_block_ptp.reshape(-1) > thresholds[spike_idx_unfold]).reshape(N, -1)
        low_threshold_accept_idx = (denoised_low_block_ptp.reshape(-1) > thresholds[spike_idx_unfold]).reshape(N, -1)

        
        
        
        top_phase_shifted = (torch.argmax(torch.swapaxes(denoised_top_low_block_wfs[:N, :, :], 0, 2) * spk_signs, 1) - offset).transpose(0,1)
        low_phase_shifted = (torch.argmax(torch.swapaxes(denoised_top_low_block_wfs[N:, :, :], 0, 2) * spk_signs, 1) - offset).transpose(0,1)

        
        
        top_phase_shifted_diff = torch.unsqueeze(top_phase_shifted, 2) - torch.unsqueeze(top_phase_shifted, 1) 
        triu_index = torch.triu_indices(block_C, block_C, 0)
        top_phase_shifted_diff[:, triu_index[0,:], triu_index[1,:]] = T
        top_phase_accept_idx = (torch.min(top_phase_shifted_diff, 2)[0] <=10)
    
    
        
        low_phase_shifted_diff = torch.unsqueeze(low_phase_shifted, 2) - torch.unsqueeze(low_phase_shifted, 1) 
        triu_index = torch.triu_indices(block_C, block_C, 0)
        low_phase_shifted_diff[:, triu_index[0,:], triu_index[1,:]] = T
        low_phase_accept_idx = (torch.min(low_phase_shifted_diff, 2)[0] <=10)
        
       
        
        top_phase_shifted = torch.where(top_threshold_accept_idx & top_phase_accept_idx, top_phase_shifted, T)
        low_phase_shifted = torch.where(low_threshold_accept_idx & low_phase_accept_idx, low_phase_shifted, T)
        
         
            
        top_phase_shifted = torch.cat((top_phase[:, None],top_phase_shifted), 1)
        low_phase_shifted = torch.cat((low_phase_shifted,low_phase[:, None]), 1)
        
        
        c = (top_phase_shifted < T)
        idx = torch.arange(0, c.shape[1])
        c2 = c * idx 
        top_phase = torch.argmax(c2, 1)
        
        
        
        c = (low_phase_shifted < T)
        idx = torch.arange(c.shape[1], 0, -1)
        c2 = c * idx
        low_phase = torch.argmax(c2, 1)
        
        # print(low_phase)
        
        top_border_phase_shift = top_phase_shifted[range(N), top_phase]
        low_border_phase_shift = low_phase_shifted[range(N), low_phase]

        phase_shift_pick_array = torch.cat((top_border_phase_shift[:,None].repeat(1, block_C), low_border_phase_shift[:,None].repeat(1, block_C)), 0).reshape(-1)
        
        
        
    
    return denoised_wfs#, phase_shift


# %%
wfs_traveler = torch.tensor(wfs_traveler)
traveler_maxchans = torch.tensor(traveler_maxchans)
denoised_wfs = block_segment_phase_shit_denoise(wfs_traveler, traveler_maxchans, torch.tensor(maxCH_neighbor), torch.tensor(top_probe_blocks), torch.tensor(low_probe_blocks), Denoiser)


# %%
denoised_wfs = denoised_wfs.detach().numpy()

# %%
traveler_maxchans = traveler_maxchans.detach().numpy()

# %%
bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
for i in range(17):
    fig, axs = plt.subplots(1, 5, figsize = (15, 8))
    axs[0].plot(wfs_traveler[i,:,:] + bias*10, 'k')
    axs[0].plot(denoised_wfs_old[i,:,:] + bias*10, 'r')
    axs[0].set_title('old denoiser')
    
    axs[1].plot(wfs_traveler[i,:,:] + bias*10, 'k')
    axs[1].plot(denoised_wfs[i,:,:40] + bias*10, 'g')
    axs[1].set_title('phase-shift denoiser')
    axs[1].vlines(42, 0, 400)
    
#     axs[1].scatter(phase_shift[i,:40] + 42, np.arange(40)*10, c = 'b')
    
#     for j in range(3):
#         top_block = top_probe_blocks[traveler_maxchans[i], j, :]
#         low_block = low_probe_blocks[traveler_maxchans[i], j, :]
#         axs[1].scatter(phase_shift[i,top_block] + 42, top_block*10)
#         axs[1].scatter(phase_shift[i,low_block] + 42, low_block*10)
    axs[2].imshow(wfs_traveler[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[2].set_title('raw wavefors')
    
    axs[3].imshow(denoised_wfs_old[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[3].set_title('old denoiser')
    
    axs[4].imshow(denoised_wfs[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[4].set_title('phase-shift denoiser')
    
    plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/segment_phaseshift_denoiser_result/traveling_spike_compare'+ '/traveling_unit' + str(i) + '_two_spit_phaseshift_denoise.png')
    # plt.plot(denoised_wfs_old[i,:,:40,0] + bias*10, 'k');
    # plt.plot(denoised_wfs_array[i,:,:] + bias*10, 'r');

# %%
