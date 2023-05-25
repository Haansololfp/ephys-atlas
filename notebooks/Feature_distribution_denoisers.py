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

# %%
one = ONE(base_url="https://alyx.internationalbrainlab.org")

# %%
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'
template_raw_wfs_benchmark = np.load(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy')

# %%
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
import h5py
h5_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_69c9a415-f7fa-4208-887b-1417c1479b48_probe_probe00_pID_1a276285-8b0e-4cc9-9f0a-a3a002978724'
h5_dir = h5_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    channel_index = h5['channel_index'][:]
    geom = h5['geom'][:]

# %%
print(torch.version.cuda)

# %%
from spike_psvae.denoise import SingleChanDenoiser
from spike_psvae import denoise
ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(channel_index, geom)
dn = SingleChanDenoiser().load().to("cuda")

# %%
from spike_ephys import cell_type_feature
from scipy import signal

# %%
device = None
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
        
        N, T, C = np.shape(unit_wfs)
        
        max_chan = max_channels[j]
        ci = channel_index[max_chan]
        max_chan_idx = np.where(ci == max_chan)[0]
        
        wfs_to_denoise = np.swapaxes(unit_wfs, 1, 2)
        
        wfs_denoised_old = dn(torch.FloatTensor(wfs_to_denoise).reshape(-1, 121)).reshape(wfs_to_denoise.shape)
        wfs_denoised_old = wfs_denoised_old.detach().numpy()
        wfs_denoised_old = np.swapaxes(wfs_denoised_old, 1, 2)
        
        maxchans = np.int32(np.ones(N)*max_chan)
        
        
        waveforms = torch.as_tensor(unit_wfs, device=device, dtype=torch.float)
        waveforms_denoise = denoise.multichan_phase_shift_denoise(waveforms, ci_graph_on_probe, torch.tensor(maxCH_neighbor).type(torch.LongTensor),  dn, maxchans = maxchans)
        wfs_denoised = waveforms_denoise.detach().numpy()
        # wfs_denoised = np.swapaxes(wfs_denoised, 1, 2)
        
        wfs_denoised_old_align = np.zeros(np.shape(wfs_denoised_old))
        wfs_denoised_align = np.zeros(np.shape(wfs_denoised))
        
        phase_shift = np.argmax(wfs_denoised_old[np.arange(N), :, np.int32(np.ones(N)*max_chan_idx)] * np.sign(template[42 , max_chan_idx]), 1) - 42
        
        for k in range(len(phase_shift)):
            ps = phase_shift[k]
            wfs_denoised_old_align[k,:,:] = np.roll(wfs_denoised_old[k, :, :], -ps, 0)
            wfs_denoised_align[k,:,:] = np.roll(wfs_denoised[k, :, :], -ps, 0)
        
        denoised_old[units[j]] = wfs_denoised_old
        denoised[units[j]] = wfs_denoised
        denoised_old_align[units[j]] = wfs_denoised_old_align
        denoised_align[units[j]] = wfs_denoised_align
        
    template_raw_wfs_benchmark[pID]['denoised_old_no_align'] = denoised_old
    template_raw_wfs_benchmark[pID]['denoised_no_align'] = denoised
    template_raw_wfs_benchmark[pID]['denoised_old'] = denoised_old_align
    template_raw_wfs_benchmark[pID]['denoised'] = denoised_align

# %% jupyter={"outputs_hidden": true}
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
    
    denoised_old_align = template_raw_wfs_benchmark[pID]['denoised_old']
    denoised_align = template_raw_wfs_benchmark[pID]['denoised']
    
    units = list(wfs.keys())
    l = len(units)
    
    
    for j in range(len(units)):
        unit_wfs = wfs[units[j]]
        template = temp[units[j]]
        unit_denoised_old = denoised_old[units[j]]
        unit_denoised = denoised[units[j]]
        
        unit_denoised_old_align = denoised_old_align[units[j]]
        unit_denoised_align = denoised_align[units[j]]
        
        ci = channel_index[max_channels[j]]
        
        not_nan_idx = np.where(ci<384)[0]
        ci_geom = geom[ci[not_nan_idx], :]
        
        maxCH_idx = np.argmax(np.ptp(template[:, not_nan_idx], 0))
        
        z_rel = np.linalg.norm(ci_geom[:, :] - ci_geom[maxCH_idx, :], axis=1)
        

        old_denoise_MSE = np.mean(np.abs(unit_denoised_old_align - template)[:,:,not_nan_idx], axis = 1)
        denoise_MSE = np.mean(np.abs(unit_denoised_align - template)[:,:,not_nan_idx], axis = 1)
        
        old_denoise_align_MSE_all.append(np.reshape(old_denoise_MSE, -1))
        phase_shift_denoise_align_MSE_all.append(np.reshape(denoise_MSE, -1))
        
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
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/multi-channel_denoise_compare'

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

plt.savefig(save_dir + '/mean_abs_error_with_respect_to_template_no_align_xz_dist.png')

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

plt.suptitle("error compared to template (blue: old, orange: new)")
plt.savefig(save_dir + "/feature_differences_compared_to_templates.png")

# %%
np.save(manually_picked_temp_dir + '/templates_w_raw_waveforms_denoise_compare.npy', template_raw_wfs_benchmark)

# %% [markdown]
# generate traveling spike

# %%
from neurowaveforms.model import generate_waveform
wfs_array = []
colliding_array = []
maxCH_array = []
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    units = list(template_raw_wfs_benchmark[pID]['temp'].keys())
    
    for j in range(len(units)):
        template =  np.array(template_raw_wfs_benchmark[pID]['temp'][units[j]])
        maxchans = np.array(template_raw_wfs_benchmark[pID]['maxchan'])
        ptps = template.ptp(0)
        maxchan = np.nanargmax(ptps)
        
        
        wfs_array.append(template[:, maxchan])
        colliding_array.append(np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])[:,:,:])
    maxCH_array.append(np.repeat(np.reshape(maxchans, -1), 100))

# %%
wfs_array = np.array(wfs_array)
colliding_array = np.array(colliding_array)
colliding_array = np.reshape(colliding_array, [-1, 121, 40])
maxCH_array = np.concatenate(maxCH_array)

# %%
i = 18
j = 24
maxCH = maxCH_array[i]
ci = channel_index[maxCH]
wav = generate_waveform(wfs_array[j,:],  sxy=np.concatenate([geom[maxCH], [0]]), wxy=np.concatenate([geom[ci], np.zeros((40,1))], axis = 1));
bias = bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
offset= 10

plt.plot(wav*5000 + bias*offset + np.roll(colliding_array[i,:,:], 25, axis = 0));

# %%
traveling_wfs = []
traveling_wfs_clean = []

traveling_wfs_denoise = []
traveling_wfs_denoise_old = []

maxchannels = []

v = 1
for i in range(10000):
    idx_0 = np.random.choice(len(wfs_array))
    idx_1 = np.random.choice(len(colliding_array))
    
    
    maxCH = maxCH_array[idx_1]
    ci = channel_index[maxCH]
    
    wav = generate_waveform(wfs_array[idx_0,:], sxy=np.concatenate([geom[maxCH], [0]]), wxy=np.concatenate([geom[ci], np.zeros((40,1))], axis = 1), vertical_velocity_mps = v)
    jitter = np.random.randint(5, 100)
    synth_wfs = wav*5000 + np.roll(colliding_array[idx_1,:,:], jitter, axis = 0)
    
    traveling_wfs.append(synth_wfs)
    traveling_wfs_clean.append(wav*5000)
    maxchannels.append(maxCH)
    

# %%
traveling_wfs = np.array(traveling_wfs)
traveling_wfs_clean = np.array(traveling_wfs_clean)
maxchannels = np.array(maxchannels)

# %% jupyter={"outputs_hidden": true}
batch_size = 100
batch_N = 100
device = None

denoised_features = []
denoised_features_old = []
clean_wfs_features = []

for i in range(batch_N):
    idx_s = i*batch_size
    idx_e = (i+1)*batch_size
    
    wfs_to_denoise = traveling_wfs[idx_s:idx_e, :, :]
    template = traveling_wfs_clean[idx_s:idx_e, :, :]
    
    wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)   
    wfs_denoised_old = dn(torch.FloatTensor(wfs_to_denoise).reshape(-1, 121)).reshape(wfs_to_denoise.shape)
    wfs_denoised_old = wfs_denoised_old.detach().numpy()
    wfs_denoised_old = np.swapaxes(wfs_denoised_old, 1, 2)   
    
    max_chan = maxchannels[idx_s: idx_e]
    
    maxchans = max_chan#np.int32(np.ones(batch_size)*max_chan)
    
    
    wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)
    waveforms = torch.as_tensor(wfs_to_denoise, device=device, dtype=torch.float)
    waveforms_denoise = denoise.multichan_phase_shift_denoise(waveforms, ci_graph_on_probe, torch.tensor(maxCH_neighbor).type(torch.LongTensor), dn, maxchans = maxchans)
    wfs_denoised = waveforms_denoise.detach().numpy()
    
    traveling_wfs_denoise_old.append(wfs_denoised_old)
    traveling_wfs_denoise.append(wfs_denoised)
    
    
    wfs_feature_values = np.zeros((batch_size, 11))
    
    waveforms = signal.resample(wfs_denoised, 1210, axis = 1)
    wfs_feature_values[:, 0] = cell_type_feature.peak_value(waveforms)
    wfs_feature_values[:, 1] = cell_type_feature.ptp_duration(waveforms)
    wfs_feature_values[:, 2] = cell_type_feature.halfpeak_duration(waveforms)
    wfs_feature_values[:, 3] = cell_type_feature.peak_trough_ratio(waveforms)

    wfs_feature_values[:, 4] = cell_type_feature.reploarization_slope(waveforms, fs*10)
    wfs_feature_values[:, 5] = cell_type_feature.recovery_slope(waveforms, fs*10)
    wfs_feature_values[:, 6] = cell_type_feature.depolarization_slope(waveforms, fs*10)

    wfs_feature_values[:, 7] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, maxchans)
    wfs_feature_values[:, 8] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, maxchans)
    wfs_feature_values[:, [9, 10]] = np.array(cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, maxchans)).T

    denoised_features.append(wfs_feature_values)

    ######
    
    wfs_feature_values = np.zeros((100, 11))

    waveforms = signal.resample(wfs_denoised_old, 1210, axis = 1)
    wfs_feature_values[:, 0] = cell_type_feature.peak_value(waveforms)
    wfs_feature_values[:, 1] = cell_type_feature.ptp_duration(waveforms)
    wfs_feature_values[:, 2] = cell_type_feature.halfpeak_duration(waveforms)
    wfs_feature_values[:, 3] = cell_type_feature.peak_trough_ratio(waveforms)

    wfs_feature_values[:, 4] = cell_type_feature.reploarization_slope(waveforms, fs*10)
    wfs_feature_values[:, 5] = cell_type_feature.recovery_slope(waveforms, fs*10)
    wfs_feature_values[:, 6] = cell_type_feature.depolarization_slope(waveforms, fs*10)

    wfs_feature_values[:, 7] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, maxchans)
    wfs_feature_values[:, 8] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, maxchans)
    wfs_feature_values[:, [9, 10]] = np.array(cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, maxchans)).T

    denoised_features_old.append(wfs_feature_values)

    ######
    
    wfs_feature_values = np.zeros((100, 11))

    waveforms = signal.resample(template, 1210, axis = 1)
    wfs_feature_values[:, 0] = cell_type_feature.peak_value(waveforms)
    wfs_feature_values[:, 1] = cell_type_feature.ptp_duration(waveforms)
    wfs_feature_values[:, 2] = cell_type_feature.halfpeak_duration(waveforms)
    wfs_feature_values[:, 3] = cell_type_feature.peak_trough_ratio(waveforms)

    wfs_feature_values[:, 4] = cell_type_feature.reploarization_slope(waveforms, fs*10)
    wfs_feature_values[:, 5] = cell_type_feature.recovery_slope(waveforms, fs*10)
    wfs_feature_values[:, 6] = cell_type_feature.depolarization_slope(waveforms, fs*10)

    wfs_feature_values[:, 7] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, maxchans)
    wfs_feature_values[:, 8] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, maxchans)
    wfs_feature_values[:, [9, 10]] = np.array(cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, maxchans)).T

    clean_wfs_features.append(wfs_feature_values)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
# denoised_features_old = np.reshape(denoised_features_old, [batch_size*batch_N, 11])
# denoised_features = np.reshape(denoised_features, [batch_size*batch_N, 11])
clean_wfs_features = np.reshape(clean_wfs_features, [-1, 11])
clean_wfs_features[:,10]

# %%
# denoised_features_old = np.reshape(denoised_features_old, [batch_size*batch_N, 11])
# denoised_features = np.reshape(denoised_features, [batch_size*batch_N, 11])
# clean_wfs_features = np.reshape(clean_wfs_features, [batch_size*batch_N, 11])
plt.hist(denoised_features[:, 10], bins = np.arange(-1.e-5, 1.e-5, 1.e-6));

# %%
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/synthesize_traveling_denoise'

denoised_features_old = np.reshape(denoised_features_old, [batch_size*batch_N, 11])
denoised_features = np.reshape(denoised_features, [batch_size*batch_N, 11])
clean_wfs_features = np.reshape(clean_wfs_features, [batch_size*batch_N, 11])

titles = ['peak_value', 'ptp_duration', 'halfpeak_duration', 'peak_trough_ratio', 'repolarization_slope', 'recovery_slope', 'depolarization_slope',
          'spatial_spread_w_threshold', 'spatial_spread_weighted_dist','velocity_above','velocity_below']

fig, axs = plt.subplots(6, 2, figsize = [6, 10], constrained_layout=True)
for j in range(11):       
    peak_diff = denoised_features_old[:, j] - clean_wfs_features[:, j]
    peak_diff_new = denoised_features[:, j] - clean_wfs_features[:, j]

    
    row = j // 2
    col = np.mod(j, 2)
    
    a = np.abs(np.reshape(peak_diff, -1))
    b = np.abs(np.reshape(peak_diff_new, -1))
    
    # a = np.reshape(peak_diff, -1)
    # b = np.reshape(peak_diff_new, -1)
    
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    
    counts, bins = np.histogram(a)
    
    axs[row][col].hist(bins[:-1], bins, weights=counts, alpha = 0.5, density = True);
    axs[row][col].hist(b, bins = bins, alpha = 0.5, density = True);
    axs[row][col].set_title(titles[j])
axs[5][1].axis('off')

plt.suptitle("error compared to template (blue: old, orange: new)")
plt.savefig(save_dir + '/speed = ' + str(v) + "/feature_differences_compared_to_templates" + "_v_"  + str(v) + ".png")

# %%
np.savez(save_dir + '/speed = ' + str(v) + "/generated_traveling_wfs_v_" + str(v) + '.npz', 
        traveling_wfs = traveling_wfs,
        traveling_wfs_clean = traveling_wfs_clean,
        traveling_wfs_denoise = traveling_wfs_denoise,
        traveling_wfs_denoise_old = traveling_wfs_denoise_old[0],
        maxchannels = maxchannels,
        denoised_features = denoised_features,
        denoised_features_old = denoised_features_old,
        clean_wfs_features = clean_wfs_features)

# %% [markdown]
# running time comparison

# %%
import time


# %% jupyter={"outputs_hidden": true}
N = [10, 100, 1000, 10000]

old_denoiser_T = []
new_denoiser_T = []

for i in range(len(N)):
    n = N[i]
    
    pick_idx = np.random.choice(10000, n)
    
    wfs_to_denoise = traveling_wfs[pick_idx, :, :]
    template = traveling_wfs_clean[pick_idx, :, :]
    
    wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)  
    t0 = time.time()
    wfs_denoised_old = dn(torch.FloatTensor(wfs_to_denoise).reshape(-1, 121)).reshape(wfs_to_denoise.shape)
    t1 = time.time()
    old_denoiser_T.append(t1 - t0)
    wfs_denoised_old = wfs_denoised_old.detach().numpy()
    wfs_denoised_old = np.swapaxes(wfs_denoised_old, 1, 2)   
    
    max_chan = maxchannels[pick_idx]
    
    maxchans = max_chan#np.int32(np.ones(batch_size)*max_chan)
    
    
    wfs_to_denoise = np.swapaxes(wfs_to_denoise, 1, 2)
    waveforms = torch.as_tensor(wfs_to_denoise, device=device, dtype=torch.float)
    t0 = time.time()
    waveforms_denoise = denoise.multichan_phase_shift_denoise(waveforms, ci_graph_on_probe, torch.tensor(maxCH_neighbor).type(torch.LongTensor), dn, maxchans = maxchans)
    t1 = time.time()
    new_denoiser_T.append(t1-t0)
    wfs_denoised = waveforms_denoise.detach().numpy()
    
    traveling_wfs_denoise_old.append(wfs_denoised_old)
    traveling_wfs_denoise.append(wfs_denoised)

# %%
plt.plot(np.array(N[0:3]), np.array(old_denoiser_T[0:3])/np.array(new_denoiser_T))
# plt.plot(np.array(N), np.array(new_denoiser_T))

# %%
plt.plot(np.array(N[0:3]), np.array(old_denoiser_T[0:3]))
plt.plot(np.array(N[0:3]), np.array(new_denoiser_T))
plt.xlabel('N')
plt.ylabel('time (s)')

# %% jupyter={"outputs_hidden": true}
bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
for i in range(1):#range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    denoised_old = template_raw_wfs_benchmark[pID]['denoised_old_no_align']
    denoised = template_raw_wfs_benchmark[pID]['denoised_no_align']
    temp = template_raw_wfs_benchmark[pID]['temp']
    
    wfs = template_raw_wfs_benchmark[pID]['wfs']
    units = list(denoised_old.keys())
    for j in range(len(units)):
        template = temp[units[j]]
        maxCH_idx = np.argmax(np.ptp(template[:, not_nan_idx], 0))
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(wfs[units[j]][0] + bias*10, c = 'k')
        axs[0].plot(denoised_old[units[j]][0,:,:] + bias*10, c = 'b')
        
        axs[1].plot(wfs[units[j]][0] + bias*10, c = 'k')
        axs[1].plot(denoised[units[j]][0,:,:] + bias*10, c = 'g')
        axs[1].plot((denoised[units[j]][0,:,:] + bias*10)[:, maxCH_idx], c = 'r')
        axs[1].vlines(42, -5, 400)

# %%
