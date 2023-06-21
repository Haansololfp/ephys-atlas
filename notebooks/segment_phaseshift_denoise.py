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
import matplotlib.pyplot as plt
from neurowaveforms.model import generate_waveform
import torch
from one.api import ONE
import h5py
from spike_psvae.denoise import SingleChanDenoiser
from spike_psvae import denoise

# %%
one = ONE(base_url="https://alyx.internationalbrainlab.org")

# %%
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'
template_raw_wfs_benchmark = np.load(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy')
template_raw_wfs_benchmark = template_raw_wfs_benchmark.item()

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
h5_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_69c9a415-f7fa-4208-887b-1417c1479b48_probe_probe00_pID_1a276285-8b0e-4cc9-9f0a-a3a002978724'
h5_dir = h5_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    channel_index = h5['channel_index'][:]
    geom = h5['geom'][:]

# %%
device= 'cuda'
maxCH_neighbor, top_probe_blocks, low_probe_blocks = denoise.probe_segment(channel_index, 40, 384)
maxCH_neighbor = torch.as_tensor(maxCH_neighbor, device=device)
top_probe_blocks = torch.as_tensor(top_probe_blocks, device=device)
low_probe_blocks = torch.as_tensor(low_probe_blocks, device=device)
Denoiser = SingleChanDenoiser().load().to(device)

# %%
denoised_wfs, phase_shift = denoise.block_segment_phase_shit_denoise(wfs_traveler, traveler_maxchans, torch.tensor(maxCH_neighbor), torch.tensor(top_probe_blocks), torch.tensor(low_probe_blocks), Denoiser)

# %% jupyter={"outputs_hidden": true}
import time
# device = 'cuda'
device = 'cuda'
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    wfs = template_raw_wfs_benchmark[pID]['wfs']
    temp = template_raw_wfs_benchmark[pID]['temp']
    max_channels = template_raw_wfs_benchmark[pID]['maxchan']
    
    units = list(wfs.keys())
    denoised_old = dict()
    denoised = dict()
    
    denoised_old_align = dict()
    denoised_align = dict()
    
    for j in range(len(units)):
        unit_wfs = wfs[units[j]]
        template = temp[units[j]]
        maxchans = torch.full((100,),max_channels[j], device=device)
        N, T, C = np.shape(unit_wfs)
        
        max_chan = max_channels[j]
        ci = channel_index[max_chan]
        max_chan_idx = np.where(ci == max_chan)[0]
        
        
        
        waveforms = torch.as_tensor(unit_wfs, device=device, dtype=torch.float)
        t0 = time.time()
        # wfs_denoised = denoise.block_segment_phase_shit_denoise(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, Denoiser, device)
        wfs_denoised = denoise.block_segment_phase_shit_denoise_radial_hallucination_suppress(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, ci_graph_all_maxCH_uniq, maxCH_neighbor_ci_pick,Denoiser, device)


        print(time.time() - t0)
        wfs_denoised = wfs_denoised.cpu().detach().numpy()
        
        wfs_to_denoise = waveforms.swapaxes(1, 2)
        wfs_denoised_old = Denoiser(wfs_to_denoise.reshape(-1, 121)).reshape(wfs_to_denoise.shape)

        wfs_denoised_old = torch.swapaxes(wfs_denoised_old, 1, 2)
        
        wfs_denoised_old = wfs_denoised_old.cpu().detach().numpy()
        

        denoised_old[units[j]] = wfs_denoised_old
        denoised[units[j]] = wfs_denoised

    template_raw_wfs_benchmark[pID]['denoised_old_no_align'] = denoised_old
    template_raw_wfs_benchmark[pID]['denoised_no_align'] = denoised


# %% jupyter={"outputs_hidden": true}
from spike_ephys import cell_type_feature
from scipy import signal

old_denoise_MSE_all = []
phase_shift_denoise_MSE_all = []
z_rel_all = []

old_denoise_align_MSE_all = []
phase_shift_denoise_align_MSE_all = []

fs = 30000

for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    wfs = template_raw_wfs_benchmark[pID]['wfs']
    temp = template_raw_wfs_benchmark[pID]['temp']
    max_channels = template_raw_wfs_benchmark[pID]['maxchan']
    denoised_old = template_raw_wfs_benchmark[pID]['denoised_old_no_align']
    denoised = template_raw_wfs_benchmark[pID]['denoised_no_align']
    
    # denoised_old_align = template_raw_wfs_benchmark[pID]['denoised_old']
    # denoised_align = template_raw_wfs_benchmark[pID]['denoised']
    
    units = list(wfs.keys())
    l = len(units)
    
    
    for j in range(len(units)):
        unit_wfs = wfs[units[j]]
        template = temp[units[j]]
        unit_denoised_old = denoised_old[units[j]][:, :, range(40)]
        unit_denoised = denoised[units[j]][:, :, range(40)]
        
        # unit_denoised_old_align = denoised_old_align[units[j]]
        # unit_denoised_align = denoised_align[units[j]]
        
        ci = channel_index[max_channels[j]]
        
        not_nan_idx = np.where(ci<384)[0]
        ci_geom = geom[ci[not_nan_idx], :]
        
        maxCH_idx = np.argmax(np.ptp(template[:, not_nan_idx], 0))
        
        z_rel = np.linalg.norm(ci_geom[:, :] - ci_geom[maxCH_idx, :], axis=1)
        

#         old_denoise_MSE = np.mean(np.abs(unit_denoised_old_align - template)[:,:,not_nan_idx], axis = 1)
#         denoise_MSE = np.mean(np.abs(unit_denoised_align - template)[:,:,not_nan_idx], axis = 1)
        
#         old_denoise_align_MSE_all.append(np.reshape(old_denoise_MSE, -1))
#         phase_shift_denoise_align_MSE_all.append(np.reshape(denoise_MSE, -1))
        
        old_denoise_MSE = np.mean(np.abs(unit_denoised_old - template)[:,:,not_nan_idx], axis = 1)
        denoise_MSE = np.mean(np.abs(unit_denoised - template)[:,:,not_nan_idx], axis = 1)
        
        old_denoise_MSE_all.append(np.reshape(old_denoise_MSE, -1))
        phase_shift_denoise_MSE_all.append(np.reshape(denoise_MSE, -1))

#         old_denoise_MSE = np.mean(np.abs((unit_denoised_old_align - template)/template)[:,:,not_nan_idx], axis = 1)
#         denoise_MSE = np.mean(np.abs((unit_denoised_align - template)/template)[:,:,not_nan_idx], axis = 1)

#         old_denoise_align_MSE_all.append(np.reshape(old_denoise_MSE, -1))
#         phase_shift_denoise_align_MSE_all.append(np.reshape(denoise_MSE, -1))

#         old_denoise_MSE = np.mean(np.abs((unit_denoised_old - template)/template)[:,:,not_nan_idx], axis = 1)
#         denoise_MSE = np.mean(np.abs((unit_denoised - template)/template)[:,:,not_nan_idx], axis = 1)

#         old_denoise_MSE_all.append(np.reshape(old_denoise_MSE, -1))
#         phase_shift_denoise_MSE_all.append(np.reshape(denoise_MSE, -1))
        
        
        z_rel_all.append(np.tile(z_rel, 100))
        
        wfs_feature_values = np.zeros((100, 11))
        
        #1) peak_value
        #2) ptp_duration
        #3) halfpeak_duration
        #4) peak_trough_ratio
        #5) repolarization_slope
        #6) recovery_slope
        #7) depolarization_slope
        #8) spatial_spread_w_threshold
        #9) spatial_spread_weighted_dist
        #10) velocity
        
        
        waveforms = signal.resample(unit_denoised, 1210, axis = 1)
        wfs_feature_values[:, 0] = cell_type_feature.peak_value(waveforms)
        wfs_feature_values[:, 1] = cell_type_feature.ptp_duration(waveforms)
        wfs_feature_values[:, 2] = cell_type_feature.halfpeak_duration(waveforms)
        wfs_feature_values[:, 3] = cell_type_feature.peak_trough_ratio(waveforms)
        
        wfs_feature_values[:, 4] = cell_type_feature.reploarization_slope(waveforms, fs*10)
        wfs_feature_values[:, 5] = cell_type_feature.recovery_slope(waveforms, fs*10)
        wfs_feature_values[:, 6] = cell_type_feature.depolarization_slope(waveforms, fs*10)
        
        wfs_feature_values[:, 7] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, np.int32(np.ones(100)*max_channels[j]))
        wfs_feature_values[:, 8] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, np.int32(np.ones(100)*max_channels[j]))
        wfs_feature_values[:, [9, 10]] = np.array(cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, np.int32(np.ones(100)*max_channels[j]))).T
        
        
        template_raw_wfs_benchmark[pID]['denoised_features'] = wfs_feature_values
        
        ######
        
        wfs_feature_values = np.zeros((100, 11))
        
        waveforms = signal.resample(unit_denoised_old, 1210, axis = 1)
        wfs_feature_values[:, 0] = cell_type_feature.peak_value(waveforms)
        wfs_feature_values[:, 1] = cell_type_feature.ptp_duration(waveforms)
        wfs_feature_values[:, 2] = cell_type_feature.halfpeak_duration(waveforms)
        wfs_feature_values[:, 3] = cell_type_feature.peak_trough_ratio(waveforms)
        
        wfs_feature_values[:, 4] = cell_type_feature.reploarization_slope(waveforms, fs*10)
        wfs_feature_values[:, 5] = cell_type_feature.recovery_slope(waveforms, fs*10)
        wfs_feature_values[:, 6] = cell_type_feature.depolarization_slope(waveforms, fs*10)
        
        wfs_feature_values[:, 7] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, np.int32(np.ones(100)*max_channels[j]))
        wfs_feature_values[:, 8] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, np.int32(np.ones(100)*max_channels[j]))
        wfs_feature_values[:, [9, 10]] = np.array(cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, np.int32(np.ones(100)*max_channels[j]))).T
        
        
        template_raw_wfs_benchmark[pID]['denoised_old_features'] = wfs_feature_values
        
        ######
        
        wfs_feature_values = np.zeros((11, ))
        
        waveforms = signal.resample(template[None,:,:], 1210, axis = 1)
        wfs_feature_values[0] = cell_type_feature.peak_value(waveforms)
        wfs_feature_values[1] = cell_type_feature.ptp_duration(waveforms)
        wfs_feature_values[2] = cell_type_feature.halfpeak_duration(waveforms)
        wfs_feature_values[3] = cell_type_feature.peak_trough_ratio(waveforms)
        
        wfs_feature_values[4] = cell_type_feature.reploarization_slope(waveforms, fs*10)
        wfs_feature_values[5] = cell_type_feature.recovery_slope(waveforms, fs*10)
        wfs_feature_values[6] = cell_type_feature.depolarization_slope(waveforms, fs*10)
        
        wfs_feature_values[7] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, np.int32(np.ones(100)*max_channels[j]))
        wfs_feature_values[8] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, np.int32(np.ones(100)*max_channels[j]))
        wfs_feature_values[[9, 10]] = np.array(cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, np.int32(np.ones(100)*max_channels[j]))).T
        
        
        template_raw_wfs_benchmark[pID]['template_features'] = wfs_feature_values

# %%
wfs_traveler = np.load('/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/wfs_traveler.npy')
traveler_maxchans = np.load('/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/wfs_traveler_maxCH.npy')

wfs_traveler = torch.tensor(wfs_traveler)
traveler_maxchans = torch.tensor(traveler_maxchans)

# %%
denoised_wfs = denoise.block_segment_phase_shit_denoise_radial_hallucination_suppress(wfs_traveler, traveler_maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, ci_graph_all_maxCH_uniq, maxCH_neighbor_ci_pick, Denoiser, device)


# %%
denoised_wfs = denoised_wfs.detach().numpy()
N, T, C = wfs_traveler.shape
denoised_wfs_old = Denoiser(wfs_traveler.swapaxes(1,2).reshape(-1, T)).reshape(N, C, T)
denoised_wfs_old = denoised_wfs_old.detach().numpy()
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
    axs[1].plot(denoised_wfs[i,:,:40] + bias*10, 'g')
    axs[1].set_title('phase-shift denoiser')
    
    axs[2].imshow(wfs_traveler[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[2].set_title('raw wavefors')
    
    axs[3].imshow(denoised_wfs_old[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[3].set_title('old denoiser')
    
    axs[4].imshow(denoised_wfs[i,:,:].T, aspect = 'auto', cmap = 'RdBu', vmin = -6, vmax = 6, origin = 'lower')
    axs[4].set_title('phase-shift denoiser')


# %%
bins = np.unique(z_rel_all)
z_rel_all = np.reshape(z_rel_all, -1)
old_denoise_MSE_all = np.reshape(old_denoise_MSE_all, -1)
phase_shift_denoise_MSE_all = np.reshape(phase_shift_denoise_MSE_all, -1)

binned_old_denoised_mean = np.zeros(len(bins))
binned_old_denoised_std = np.zeros(len(bins))

phaseshift_denoised_mean = np.zeros(len(bins))
phaseshift_denoised_std = np.zeros(len(bins))

for i in range(len(bins)):
    which = np.where(z_rel_all == bins[i])[0]
    
    binned_old_denoised_mean[i] = np.mean(old_denoise_MSE_all[which])
    binned_old_denoised_std[i] = np.std(old_denoise_MSE_all[which])
    
    phaseshift_denoised_mean[i] = np.mean(phase_shift_denoise_MSE_all[which])
    phaseshift_denoised_std[i] = np.std(phase_shift_denoise_MSE_all[which])

plt.plot(bins, binned_old_denoised_mean, 'b-', label = 'single channel denoiser')
fill_1 = plt.fill_between(bins, binned_old_denoised_mean - binned_old_denoised_std, binned_old_denoised_mean + binned_old_denoised_std, color='b', alpha=0.2)

plt.plot(bins, phaseshift_denoised_mean, 'r-', label = 'phase-shift single channel denoiser w hallucination suppress')
fill_1 = plt.fill_between(bins, phaseshift_denoised_mean - phaseshift_denoised_std, phaseshift_denoised_mean + phaseshift_denoised_std, color='r', alpha=0.2)

plt.xlabel('dist to maxchan(micron)')
plt.ylabel('denoised waveform mean error')

plt.legend()

# plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/segment_phaseshift_denoiser_result' + '/err_dist_plot_multichannel_spikes.png')

# %%
template_raw_wfs_benchmark = np.load('/moto/stats/users/hy2562/projects/ephys_atlas/segment_phaseshift_denoiser_result'+ '/segments_phaseshift_templates_w_raw_waveforms_denoise_compare.npy')
template_raw_wfs_benchmark = template_raw_wfs_benchmark.item()

# %%
titles = ['peak_value', 'ptp_duration', 'halfpeak_duration', 'peak_trough_ratio', 'repolarization_slope', 'recovery_slope', 'depolarization_slope',
          'spatial_spread_w_threshold', 'spatial_spread_weighted_dist','velocity_above','velocity_below']

fig, axs = plt.subplots(6, 2, figsize = [6, 10], constrained_layout=True)
for j in range(11):       
    peak_diff = []
    peak_diff_new = []

    for i in range(len(Benchmark_pids)):
        pID = Benchmark_pids[i]
        wfs_feature_values = template_raw_wfs_benchmark[pID]['denoised_old_features']
        wfs_new_feature_values = template_raw_wfs_benchmark[pID]['denoised_features']
        temp_feature_values = template_raw_wfs_benchmark[pID]['template_features']
        peak_diff.append(np.reshape(wfs_feature_values[:, j] - temp_feature_values[j], -1))
        peak_diff_new.append(np.reshape(wfs_new_feature_values[:, j] - temp_feature_values[j], -1))
    
    row = j // 2
    col = np.mod(j, 2)
    
    a = np.abs(np.reshape(peak_diff, -1))
    b = np.abs(np.reshape(peak_diff_new, -1))
    
    # a = np.reshape(peak_diff, -1)
    # b = np.reshape(peak_diff_new, -1)
    
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if (j == 9) | (j == 10):
        axs[row][col].hist(a, bins = np.arange(0, 1.e-5, 1.e-6), alpha = 0.5, density = True);
        axs[row][col].hist(b, bins = np.arange(0, 1.e-5, 1.e-6), alpha = 0.5, density = True);
        axs[row][col].set_title(titles[j])
    else:
        
        counts, bins = np.histogram(a)

        axs[row][col].hist(bins[:-1], bins, weights=counts, alpha = 0.5, density = True);
        axs[row][col].hist(b, bins = bins, alpha = 0.5, density = True);
        axs[row][col].set_title(titles[j])
axs[5][1].axis('off')


# plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/segment_phaseshift_denoiser_result' + '/features_difference.png')

# %%
np.save( '/moto/stats/users/hy2562/projects/ephys_atlas/segment_phaseshift_denoiser_result'+ '/segments_phaseshift_templates_w_raw_waveforms_denoise_compare.npy', template_raw_wfs_benchmark)

# %%
torch.cuda.empty_cache()

# %%
import time

# %%
opt_block_segment_phase_shit_denoise = torch.compile(denoise.block_segment_phase_shit_denoise)

# %%
from spike_psvae.denoise import SingleChanDenoiser
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/synthesize_traveling_denoise'
v = 1
data = np.load(save_dir + '/speed = ' + str(v) + "/generated_traveling_wfs_v_" + str(v) + '.npz')
traveling_wfs = data['traveling_wfs']
maxchannels = data['maxchannels']
N = [20000, 10000, 1000, 100, 10]
T = 5
old_denoiser_T = np.zeros((T, len(N)))
new_denoiser_T = np.zeros((T, len(N)))

device = 'cuda'
# ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(channel_index, geom, device)
# ci_graph_all_maxCH_uniq = denoise.make_ci_graph_all_maxCH(ci_graph_on_probe, maxCH_neighbor, device)
# Denoiser = SingleChanDenoiser().load().to(device)
# dn = SingleChanDenoiser().load().to(device)
for j in range(T):
    for i in range(len(N)):
        torch.cuda.empty_cache()
        n = N[i]

        pick_idx = np.random.choice(10000, n)

        wfs_to_denoise = traveling_wfs[pick_idx, :, :]

        wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)  
        waveforms = torch.FloatTensor(wfs_to_denoise).to(device).reshape(-1, 121)
        t0 = time.time()
        wfs_denoised_old = Denoiser(waveforms).reshape(wfs_to_denoise.shape)
        t1 = time.time()
        
        del(wfs_denoised_old)
        
        old_denoiser_T[j, i] = t1 - t0
        # old_denoiser_T.append(t1 - t0)
        # wfs_denoised_old = wfs_denoised_old.to('cpu').detach().numpy()
        # wfs_denoised_old = np.swapaxes(wfs_denoised_old, 1, 2)   

        max_chan = maxchannels[pick_idx]

        maxchans = torch.tensor(max_chan, device=device)#np.int32(np.ones(batch_size)*max_chan)

        # ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(channel_index, geom, device)

        wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)
        waveforms = torch.as_tensor(wfs_to_denoise, device=device, dtype=torch.float)
        t0 = time.time()
        # waveforms_denoise = denoise.block_segment_phase_shit_denoise(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, Denoiser, device)
        waveforms_denoise = opt_block_segment_phase_shit_denoise(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, Denoiser, device)

        t1 = time.time()
        new_denoiser_T[j, i] = t1 - t0
        
        torch.cuda.empty_cache()
        del(waveforms_denoise)
        # new_denoiser_T.append(t1-t0)
        # wfs_denoised = waveforms_denoise.to('cpu').detach().numpy()

        # traveling_wfs_denoise_old.append(wfs_denoised_old)
        # traveling_wfs_denoise.append(wfs_denoised)

# %%
plt.scatter(np.repeat(np.array(N)[:,None], 3,  1), old_denoiser_T[2:5,:].T, c = 'b');
plt.scatter(np.repeat(np.array(N)[:,None], 3, 1), new_denoiser_T[2:5,:].T, c = 'g');
plt.plot(N, np.mean(old_denoiser_T[2:5,:], 0), c = 'b', label='single channel');
plt.plot(N, np.mean(new_denoiser_T[2:5,:], 0), c = 'g', label='segment gpurized phase-shift');

# fill_1 = plt.fill_between(N, np.mean(old_denoiser_T, 0) - np.std(old_denoiser_T, 0), np.mean(old_denoiser_T, 0) + np.std(old_denoiser_T, 0), color='b', alpha=0.2)
# fill_2 = plt.fill_between(N, np.mean(new_denoiser_T, 0) - np.std(new_denoiser_T, 0), np.mean(new_denoiser_T, 0) + np.std(new_denoiser_T, 0), color='g', alpha=0.2)

plt.legend()

plt.xlabel('N')
plt.ylabel('time(s)')

# plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/segment_phaseshift_denoiser_result' + '/runtime_compare.png')
# plt.ylim([0,1])

# %%
plt.plot(N, np.mean(new_denoiser_T[2:5,:], 0)/np.mean(old_denoiser_T[2:5,:], 0))

plt.xlabel('N')
plt.ylabel('probe segment/single channel')

# %%

# %%
import time

# %%
device = 'cuda'
# maxCH_neighbor, top_probe_blocks, low_probe_blocks = denoise.probe_segment(channel_index, 40, 384)
# maxCH_neighbor = torch.as_tensor(maxCH_neighbor, device=device)
ci_graph_on_probe, maxCH_neighbor_ci_pick = denoise.make_ci_graph(channel_index, geom, device)
ci_graph_all_maxCH_uniq = denoise.make_ci_graph_all_maxCH(ci_graph_on_probe, maxCH_neighbor_ci_pick, device)
Denoiser = SingleChanDenoiser().load().to(device)

# %%
top_probe_blocks = torch.as_tensor(top_probe_blocks, device=device)
low_probe_blocks = torch.as_tensor(low_probe_blocks, device=device)

# %%
opt_block_segment_phase_shit_denoise_radial_hallucination_suppress = torch.compile(denoise.block_segment_phase_shit_denoise_radial_hallucination_suppress)

# %%
from spike_psvae.denoise import SingleChanDenoiser
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/synthesize_traveling_denoise'
v = 1
data = np.load(save_dir + '/speed = ' + str(v) + "/generated_traveling_wfs_v_" + str(v) + '.npz')
traveling_wfs = data['traveling_wfs']
maxchannels = data['maxchannels']
N = [20000, 10000, 1000, 100, 10]
T = 5
old_denoiser_T = np.zeros((T, len(N)))
new_denoiser_T = np.zeros((T, len(N)))

device = 'cuda'
# ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(channel_index, geom, device)
# ci_graph_all_maxCH_uniq = denoise.make_ci_graph_all_maxCH(ci_graph_on_probe, maxCH_neighbor, device)
# Denoiser = SingleChanDenoiser().load().to(device)
# dn = SingleChanDenoiser().load().to(device)
for j in range(T):
    for i in range(len(N)):
        torch.cuda.empty_cache()
        n = N[i]

        pick_idx = np.random.choice(10000, n)

        wfs_to_denoise = traveling_wfs[pick_idx, :, :]

        wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)  
        waveforms = torch.FloatTensor(wfs_to_denoise).to(device).reshape(-1, 121)
        t0 = time.time()
        wfs_denoised_old = Denoiser(waveforms).reshape(wfs_to_denoise.shape)
        t1 = time.time()
        
        del(wfs_denoised_old)
        
        old_denoiser_T[j, i] = t1 - t0
        # old_denoiser_T.append(t1 - t0)
        # wfs_denoised_old = wfs_denoised_old.to('cpu').detach().numpy()
        # wfs_denoised_old = np.swapaxes(wfs_denoised_old, 1, 2)   

        max_chan = maxchannels[pick_idx]

        maxchans = torch.tensor(max_chan, device=device)#np.int32(np.ones(batch_size)*max_chan)

        # ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(channel_index, geom, device)

        wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)
        waveforms = torch.as_tensor(wfs_to_denoise, device=device, dtype=torch.float)
        t0 = time.time()
        # waveforms_denoise = denoise.block_segment_phase_shit_denoise(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, Denoiser, device)
        waveforms_denoise = denoise.block_segment_phase_shit_denoise_radial_hallucination_suppress(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, ci_graph_all_maxCH_uniq, maxCH_neighbor_ci_pick,Denoiser, device)

        t1 = time.time()
        new_denoiser_T[j, i] = t1 - t0
        
        torch.cuda.empty_cache()
        del(waveforms_denoise)
        # new_denoiser_T.append(t1-t0)
        # wfs_denoised = waveforms_denoise.to('cpu').detach().numpy()

        # traveling_wfs_denoise_old.append(wfs_denoised_old)
        # traveling_wfs_denoise.append(wfs_denoised)

# %%
plt.scatter(np.repeat(np.array(N)[:,None], 3,  1), old_denoiser_T[2:5,:].T, c = 'b');
plt.scatter(np.repeat(np.array(N)[:,None], 3, 1), new_denoiser_T[2:5,:].T, c = 'g');
plt.plot(N, np.mean(old_denoiser_T[2:5,:], 0), c = 'b', label='single channel');
plt.plot(N, np.mean(new_denoiser_T[2:5,:], 0), c = 'g', label='segment gpurized phase-shift');


plt.legend()

plt.xlabel('N')
plt.ylabel('time(s)')


# %%
plt.plot(N, np.mean(new_denoiser_T[2:5,:], 0)/np.mean(old_denoiser_T[2:5,:], 0))

plt.xlabel('N')
plt.ylabel('probe segment/single channel')

# %%

# %%

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
import h5py
import re

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
import numpy as np
# modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'
template_raw_wfs_benchmark = np.load(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy')
template_raw_wfs_benchmark = template_raw_wfs_benchmark.item()

# %%
bias = bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
offset= 10

save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/segment_phaseshift_denoiser_result/multi-channel_denoise_compare/'
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    
    units = list(template_raw_wfs_benchmark[pID]['temp'].keys())
    
    for j in range(len(units)):
        
        raw_wfs = np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])
        template =  np.array(template_raw_wfs_benchmark[pID]['temp'][units[j]])

        ptps = template.ptp(1)
        maxchan = np.nanargmax(ptps)
        maxchans = np.ones(np.shape(raw_wfs)[0])*maxchan

        waveforms = torch.as_tensor(raw_wfs, device=device, dtype=torch.float)
        waveforms = denoise.block_segment_phase_shit_denoise(waveforms, maxchans, maxCH_neighbor, top_probe_blocks, low_probe_blocks, Denoiser, device)
        
        waveforms =  waveforms.cpu().detach().numpy()
        
        wfs = torch.as_tensor(raw_wfs, device=device, dtype=torch.float).swapaxes(1, 2)
        wfs_denoised_old = Denoiser(wfs.reshape(-1, 121)).reshape(wfs.shape)
        wfs_denoised_old = wfs_denoised_old.cpu().detach().numpy()
        wfs_denoised_old = np.swapaxes(wfs_denoised_old, 1, 2)
        
        for k in range(5):

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
