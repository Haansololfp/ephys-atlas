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
from spike_psvae import cell_type_feature
from tqdm import tqdm
import h5py
import numpy as np
from scipy import signal
from one.api import ONE
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
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    h5_path = out_dir + '/' + 'subtraction.h5'
    batch_size = 10000
    fs = 30000
    
    with h5py.File(h5_path) as h5:
        spike_idx = h5["spike_index"][:]
        geom = h5["geom"][:]
        channel_index = h5["channel_index"][:]


    spike_num = len(spike_idx)
    h5 = h5py.File(h5_path)
    batch_n = int(np.floor(spike_num/batch_size))

    ptp_duration = np.zeros((spike_num,))
    halfpeak_duration = np.zeros((spike_num,))
    peak_trough_ratio = np.zeros((spike_num,))
    spatial_non_threshold = np.zeros((spike_num,))
    reploarization_slope = np.zeros((spike_num,))
    recovery_slope = np.zeros((spike_num,))
    

    ci_err = np.zeros((spike_num,))

    for j in tqdm(range(batch_n)):
        start_idx = j * batch_size
        end_idx = (j + 1) * batch_size
        waveforms = h5["denoised_waveforms"][start_idx:end_idx]
        spk_idx = spike_idx[start_idx:end_idx, 1]

        waveforms = signal.resample(waveforms, 1210, axis = 1)
        ptp_duration[start_idx:end_idx] = cell_type_feature.ptp_duration(waveforms)
        halfpeak_duration[start_idx:end_idx] = cell_type_feature.halfpeak_duration(waveforms)
        peak_trough_ratio[start_idx:end_idx] = cell_type_feature.peak_trough_ratio(waveforms)
        spatial_non_threshold[start_idx:end_idx] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, spk_idx)
        reploarization_slope[start_idx:end_idx] = cell_type_feature.reploarization_slope(waveforms, fs*10)
        recovery_slope[start_idx:end_idx] = cell_type_feature.recovery_slope(waveforms, fs*10)
        
        

    start_idx = batch_n * batch_size
    end_idx = None
    waveforms = h5["denoised_waveforms"][start_idx:end_idx]
    spk_idx = spike_idx[start_idx:end_idx, 1]
    
    waveforms = signal.resample(waveforms, 1210, axis = 1)
    ptp_duration[start_idx:end_idx] = cell_type_feature.ptp_duration(waveforms)
    halfpeak_duration[start_idx:end_idx] = cell_type_feature.halfpeak_duration(waveforms)
    peak_trough_ratio[start_idx:end_idx] = cell_type_feature.peak_trough_ratio(waveforms)
    spatial_non_threshold[start_idx:end_idx] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, spk_idx)
    reploarization_slope[start_idx:end_idx] = cell_type_feature.reploarization_slope(waveforms, fs*10)
    recovery_slope[start_idx:end_idx] = cell_type_feature.recovery_slope(waveforms, fs*10)
    

    np.save(out_dir + '/ptp_duration.npy', ptp_duration)
    np.save(out_dir + '/halfpeak_duration.npy', halfpeak_duration)
    np.save(out_dir + '/peak_trough_ratio.npy', peak_trough_ratio)
    np.save(out_dir + '/non_threshold_spatial_spread.npy', spatial_non_threshold)
    np.save(out_dir + '/recovery_slope.npy', recovery_slope)
    np.save(out_dir + '/reploarization_slope_window_50.npy', reploarization_slope)
    h5.close()


# %%
from neurodsp.waveforms import peak_trough_tip
from neurodsp.waveforms import plot_peaktiptrough

# %%
i = 0
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
pID = Benchmark_pids[i]
eID, probe = one.pid2eid(pID)
out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
h5_path = out_dir + '/' + 'subtraction.h5'
batch_size = 10000
fs = 30000

with h5py.File(h5_path) as h5:
    spike_idx = h5["spike_index"][:]
    geom = h5["geom"][:]
    channel_index = h5["channel_index"][:]
    
spike_num = len(spike_idx)
h5 = h5py.File(h5_path)

start_idx = i * batch_size
end_idx = (i + 1) * batch_size
waveforms = h5["denoised_waveforms"][start_idx:end_idx]

# %%
# import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
plot_peaktiptrough(df, waveforms, ax)

# %%
df = peak_trough_tip(waveforms)

# %%
waveforms = signal.resample(waveforms, 1210, axis = 1)
cell_type_feature.recovery_slope(waveforms, 300000)

# %%
spk_idx = spike_idx[start_idx:end_idx, 1]
cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, spk_idx)

# %%
plt.hist(np.nanargmax(waveforms.ptp(1),axis =1), bins = np.arange(5.5, 25.5))

# %%
mcs = np.nanargmax(waveforms.ptp(1),axis =1)

# %%
np.where((mcs!=19) & (mcs!=20))

# %%
