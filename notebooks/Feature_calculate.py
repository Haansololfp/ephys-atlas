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


# %%
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
# main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets'
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    # out_dir = main_dir + '/' + eID + '_' + probe
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    h5_path = out_dir + '/' + 'subtraction.h5'
    batch_size = 10000
    fs = 30000
    try:
        with h5py.File(h5_path) as h5:
            spike_idx = h5["spike_index"][:]
            geom = h5["geom"][:]
            channel_index = h5["channel_index"][:]
    except:
        continue

    spike_num = len(spike_idx)
    h5 = h5py.File(h5_path)
    batch_n = int(np.floor(spike_num/batch_size))

    peak_value = np.zeros((spike_num,))
    ptp_duration = np.zeros((spike_num,))
    halfpeak_duration = np.zeros((spike_num,))
    peak_trough_ratio = np.zeros((spike_num,))
    spatial_spread = np.zeros((spike_num,))
    # velocity = np.zeros((spike_num,))
    spatial_non_threshold = np.zeros((spike_num,))
    reploarization_slope = np.zeros((spike_num,))
    recovery_slope = np.zeros((spike_num,))
    

    ci_err = np.zeros((spike_num,))

    for i in tqdm(range(batch_n)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        waveforms = h5["denoised_waveforms"][start_idx:end_idx]
        spk_idx = spike_idx[start_idx:end_idx, 1]

        waveforms = signal.resample(waveforms, 1210, axis = 1)
        peak_value[start_idx:end_idx] = cell_type_feature.peak_value(waveforms)
        ptp_duration[start_idx:end_idx] = cell_type_feature.ptp_duration(waveforms)
        halfpeak_duration[start_idx:end_idx] = cell_type_feature.halfpeak_duration(waveforms)
        peak_trough_ratio[start_idx:end_idx] = cell_type_feature.peak_trough_ratio(waveforms)
        spatial_spread[start_idx:end_idx] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, spk_idx)
        
        spatial_non_threshold[start_idx:end_idx] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, spk_idx)
        reploarization_slope[start_idx:end_idx] = cell_type_feature.reploarization_slope(waveforms, fs*10)
        recovery_slope[start_idx:end_idx] = cell_type_feature.recovery_slope(waveforms, fs*10)
        
        
        # v, ci = cell_type_feature.velocity(waveforms, geom, channel_index, n_workers=64 )
        # velocity[i * batch_size: (i + 1) * batch_size] = v
        # ci_err[i * batch_size: (i + 1) * batch_size] = ci

    start_idx = batch_n * batch_size
    end_idx = None
    waveforms = h5["denoised_waveforms"][start_idx:end_idx]
    spk_idx = spike_idx[start_idx:end_idx, 1]
    
    waveforms = signal.resample(waveforms, 1210, axis = 1)
    peak_value[start_idx:end_idx] = cell_type_feature.peak_value(waveforms)
    ptp_duration[start_idx:end_idx] = cell_type_feature.ptp_duration(waveforms)
    halfpeak_duration[start_idx:end_idx] = cell_type_feature.halfpeak_duration(waveforms)
    peak_trough_ratio[start_idx:end_idx] = cell_type_feature.peak_trough_ratio(waveforms)
    spatial_spread[start_idx:end_idx] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, spk_idx)
    spatial_non_threshold[start_idx:end_idx] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, spk_idx)
    reploarization_slope[start_idx:end_idx] = cell_type_feature.reploarization_slope(waveforms, fs*10)
    recovery_slope[start_idx:end_idx] = cell_type_feature.recovery_slope(waveforms, fs*10)
    
    v, ci = cell_type_feature.velocity(waveforms, geom, channel_index, n_workers=64 )
    velocity[start_idx:end_idx] = v
    np.save(out_dir + '/ptp_duration.npy', ptp_duration)
    np.save(out_dir + '/halfpeak_duration.npy', halfpeak_duration)
    np.save(out_dir + '/peak_trough_ratio.npy', peak_trough_ratio)
    np.save(out_dir + '/spatial_spread.npy', spatial_spread)
    # np.save(out_dir + '/velocity_th_25.npy', velocity)
    # np.save(out_dir + '/velocity_ci_th_25.npy', ci_err)
    np.save(out_dir + '/non_threshold_spatial_spread.npy', spatial_non_threshold)
    np.save(out_dir + '/recovery_slope.npy', recovery_slope)
    np.save(out_dir + '/reploarization_slope_window_50.npy', reploarization_slope)
    np.save(out_dir + '/spatial_spread_th12.npy', spatial_spread)
    h5.close()


# %% jupyter={"outputs_hidden": true}
for i in range(2,len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    h5_path =out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        maxptps = h5['maxptps'][:]
    which = maxptps>6
    
    spatial_spread = np.load(out_dir + '/spatial_spread_th12.npy')
    spatial_non_threshold = np.load(out_dir + '/non_threshold_spatial_spread.npy')
    
    fig, axs = plt.subplots(1,3, figsize = (60,20))
    axs[0].hist2d(spatial_spread[which], spatial_non_threshold[which], [np.arange(0, 120, 3), np.arange(40,110,1)])
    axs[1].hist2d(maxptps[which], spatial_non_threshold[which], [np.arange(6, 14, 0.1), np.arange(40,110,1)])
    axs[2].hist2d(maxptps[which], spatial_spread[which], [np.arange(6, 14, 0.1), np.arange(0, 120, 3)])
    
    axs[0].set_xlabel('spatial_spread')
    axs[0].set_ylabel('spatial_non_threshold')
    
    axs[1].set_xlabel('maxptps')
    axs[1].set_ylabel('spatial_non_threshold')    
    
    axs[2].set_xlabel('maxptps')
    axs[2].set_ylabel('spatial_spread')
    
    plt.savefig(out_dir + '/spatial_spread_' + pID + '.png')

# %%
ptp_duration = np.load(out_dir + '/ptp_duration.npy')
peak_trough_ratio = np.load(out_dir + '/peak_trough_ratio.npy')
spatial_spread = np.load(out_dir + '/spatial_spread_th12.npy')

reploarization_slope = np.load(out_dir + '/reploarization_slope.npy')
recovery_slope = np.load(out_dir + '/recovery_slope.npy')


# %%
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable



fig, axs1 = plt.subplots(1, 7, figsize = (24, 20))


with h5py.File(h5_path) as h5:
    maxptps = h5['maxptps'][:]
    z_reg = h5['localizations'][:,2]
    x = h5['localizations'][:,0]
which = maxptps>6

bin_size = 40
z_reg_bins = np.arange(0, 3820, bin_size)
###

cmps = np.clip(ptp_duration[which]/300, 0.45, 0.6)

bin_means, bin_edges, binnumber = stats.binned_statistic(z_reg[which], ptp_duration[which]/300, statistic='mean', bins=z_reg_bins)
ptp_duration_bin_means = gaussian_filter(bin_means, sigma=3)

nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
im1 = axs1[0].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
axs1[0].set_title('ptp_duration')
axs1[0].set_xlim([-60, 100])
axs1[0].set_yticklabels([])

axs1_divider = make_axes_locatable(axs1[0])
caxs1 = axs1_divider.append_axes("right", size="7%", pad="2%")
cbs1 = fig.colorbar(im1, cax=caxs1)

###

cmps = np.clip(peak_trough_ratio[which], -1.2, 1.2)

bin_means, bin_edges, binnumber = stats.binned_statistic(z_reg[which], peak_trough_ratio[which], statistic='mean', bins=z_reg_bins)
peak_trough_ratio_bin_means = gaussian_filter(bin_means, sigma=3)

nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
im2 = axs1[1].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['RdBu'], rasterized=True)
axs1[1].set_title('peak_trough_ratio')
axs1[1].set_xlim([-60, 100])
axs1[1].set_yticklabels([])

axs2_divider = make_axes_locatable(axs1[1])
caxs2 = axs2_divider.append_axes("right", size="7%", pad="2%")
cbs2 = fig.colorbar(im2, cax=caxs2)


###
cmps = np.clip(spatial_spread[which], 0, 60)

bin_means, bin_edges, binnumber = stats.binned_statistic(z_reg[which], spatial_spread[which], statistic='mean', bins=z_reg_bins)
spatial_spread_bin_means = gaussian_filter(bin_means, sigma=3)

nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
im3 = axs1[2].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
axs1[2].set_title('spatial_spread')
axs1[2].set_xlim([-60, 100])
axs1[2].set_yticklabels([])

axs3_divider = make_axes_locatable(axs1[2])
caxs3 = axs3_divider.append_axes("right", size="7%", pad="2%")
cbs3 = fig.colorbar(im3, cax=caxs3)


###

cmps = np.clip(maxptps[which], 6, 14)

bin_means, bin_edges, binnumber = stats.binned_statistic(z_reg[which], maxptps[which], statistic='mean', bins=z_reg_bins)
maxptps_bin_means = gaussian_filter(bin_means, sigma=3)

nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
im4 = axs1[3].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
axs1[3].set_title('maxptps')
axs1[3].set_xlim([-60, 100])
axs1[3].set_yticklabels([])

axs4_divider = make_axes_locatable(axs1[3])
caxs4 = axs4_divider.append_axes("right", size="7%", pad="2%")
cbs4 = fig.colorbar(im4, cax=caxs4)


###

cmps = np.clip(spatial_non_threshold[which], 60, 90)

bin_means, bin_edges, binnumber = stats.binned_statistic(z_reg[which], spatial_non_threshold[which], statistic='mean', bins=z_reg_bins)
spatial_non_threshold_bin_means = gaussian_filter(bin_means, sigma=3)

nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
im5 = axs1[4].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
axs1[4].set_title('weighted_dist_sum')
axs1[4].set_xlim([-60, 100])
axs1[4].set_yticklabels([])

axs5_divider = make_axes_locatable(axs1[4])
caxs5 = axs5_divider.append_axes("right", size="7%", pad="2%")
cbs5 = fig.colorbar(im5, cax=caxs5)

###


cmps = np.clip(reploarization_slope[which], 10, 100)

bin_means, bin_edges, binnumber = stats.binned_statistic(z_reg[which], spatial_non_threshold[which], statistic='mean', bins=z_reg_bins)
spatial_non_threshold_bin_means = gaussian_filter(bin_means, sigma=3)

nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
im6 = axs1[5].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
axs1[5].set_title('repolarization_slope')
axs1[5].set_xlim([-60, 100])
axs1[5].set_yticklabels([])

axs6_divider = make_axes_locatable(axs1[5])
caxs6 = axs6_divider.append_axes("right", size="7%", pad="2%")
cbs6 = fig.colorbar(im6, cax=caxs6)

###

cmps = np.clip(recovery_slope[which], -15, -1)

bin_means, bin_edges, binnumber = stats.binned_statistic(z_reg[which], recovery_slope[which], statistic='mean', bins=z_reg_bins)
recovery_slope_bin_means = gaussian_filter(bin_means, sigma=3)

nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
im7 = axs1[6].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
axs1[6].set_title('recovery_slope')
axs1[6].set_xlim([-60, 100])
axs1[6].set_yticklabels([])

axs7_divider = make_axes_locatable(axs1[6])
caxs7 = axs7_divider.append_axes("right", size="7%", pad="2%")
cbs7 = fig.colorbar(im7, cax=caxs7)


# %%
plt.hist(recovery_slope, bins = np.sort(-np.arange(40)));

# %%
plt.hist(reploarization_slope,bins = np.sort(np.arange(200)));

# %%
plt.hist(spatial_non_threshold, bins = np.arange(110));

# %%
spatial_spread_2 = np.load(out_dir + '/spatial_spread_th12.npy')

# %%
plt.hist(spatial_spread, bins = np.arange(0, 120, 3));

# %% jupyter={"outputs_hidden": true}
data.values()

# %% jupyter={"outputs_hidden": true}
import pandas as pd
import seaborn as sns
from pathlib import Path
from ibllib.atlas import BrainRegions

for i in range(len(Benchmark_pids)):
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
    h5_path =out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        z_reg = h5['localizations'][:,2]
        max_ptps = h5['maxptps'][:]


    which = max_ptps>6 

    ptp_duration = np.load(out_dir + '/ptp_duration.npy')
    peak_trough_ratio = np.load(out_dir + '/peak_trough_ratio.npy')
    spatial_spread = np.load(out_dir + '/spatial_spread_th12.npy')
    reploarization_slope = np.load(out_dir + '/reploarization_slope_window_50.npy')
    recovery_slope = np.load(out_dir + '/recovery_slope.npy')
    spatial_non_threshold = np.load(out_dir + '/non_threshold_spatial_spread.npy')

    ptp_duration = ptp_duration[which]/300
    peak_trough_ratio = peak_trough_ratio[which]
    spatial_spread = spatial_spread[which]
    reploarization_slope = reploarization_slope[which]
    recovery_slope = recovery_slope[which]
    spatial_non_threshold = spatial_non_threshold[which]
    max_ptps = max_ptps[which]

    regional_ptp_duration = dict()
    regional_peak_trough_ratio = dict()
    regional_spatial_spread = dict()
    regional_reploarization_slope = dict()
    regional_recovery_slope = dict()
    regional_spatial_non_threshold = dict()
    regional_maxptps = dict()

    for j in range(len(region_labels)):
        boundary_min = boundaries[j]
        boundary_max = boundaries[j+1]

        reg_label = region_labels[j][1]
        region_which = (z_reg[which] > boundary_min) & (z_reg[which] <= boundary_max)
        
        if np.sum(region_which)>0:
            d  = ptp_duration[region_which]
            lower = np.percentile(d, 2.5)
            upper = np.percentile(d, 97.5)
            regional_ptp_duration[reg_label] = d[(d>lower) & (d<upper)]

            d  = peak_trough_ratio[region_which]
            lower = np.percentile(d, 2.5)
            upper = np.percentile(d, 97.5)
            regional_peak_trough_ratio[reg_label] = d[(d>lower) & (d<upper)]

            d  = spatial_spread[region_which]
            lower = np.percentile(d, 2.5)
            upper = np.percentile(d, 97.5)
            regional_spatial_spread[reg_label] = d[(d>lower) & (d<upper)]


            d = reploarization_slope[region_which]
            lower = np.percentile(d, 2.5)
            upper = np.percentile(d, 97.5)
            regional_reploarization_slope[reg_label] = d[(d>lower) & (d<upper)]

            d = recovery_slope[region_which]
            lower = np.percentile(d, 2.5)
            upper = np.percentile(d, 97.5)
            regional_recovery_slope[reg_label] = d[(d>lower) & (d<upper)]

            d = spatial_non_threshold[region_which]
            lower = np.percentile(d, 2.5)
            upper = np.percentile(d, 97.5)
            regional_spatial_non_threshold[reg_label] = d[(d>lower) & (d<upper)]

            d = max_ptps[region_which]
            lower = np.percentile(d, 2.5)
            upper = np.percentile(d, 97.5)
            regional_maxptps[reg_label] = d[(d>lower) & (d<upper)]
            
        else:
            regional_ptp_duration[reg_label] = np.zeros(1)*np.nan
            regional_peak_trough_ratio[reg_label] = np.zeros(1)*np.nan
            regional_spatial_spread[reg_label] = np.zeros(1)*np.nan
            regional_reploarization_slope[reg_label] = np.zeros(1)*np.nan
            regional_recovery_slope[reg_label] = np.zeros(1)*np.nan
            regional_spatial_non_threshold[reg_label] =np.zeros(1)*np.nan
            regional_maxptps[reg_label] = np.zeros(1)*np.nan

    fig, axs= plt.subplots(6,1,figsize = (40, 50)) 

    data = regional_ptp_duration
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[0], showextrema=False)
    axs[0].set_ylabel('ptp_duration')


    data = regional_peak_trough_ratio
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[1], showextrema=False)
    axs[1].set_ylabel('peak_trough_ratio')

    # data = regional_spatial_spread
    # maxsize = max([a.size for a in data.values()])
    # data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    # df = pd.DataFrame(data_pad)
    # sns.violinplot(data=df, ax = axs[2], showextrema=False)
    # axs[2].set_ylabel('spatial_spread')


    data = regional_reploarization_slope
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[2], showextrema=False)
    axs[2].set_ylabel('reploarization_slope')

    data = regional_recovery_slope
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[3], showextrema=False)
    axs[3].set_ylabel('recovery_slope')

    data = regional_spatial_non_threshold
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[4], showextrema=False)
    axs[4].set_ylabel('spatial_non_threshold')

    data = regional_maxptps
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[5], showextrema=False)
    axs[5].set_ylabel('max_ptps')


    plt.savefig(main_dir + '/feature_distribution_each_region_pID_' + pID + '.png')

# %%
