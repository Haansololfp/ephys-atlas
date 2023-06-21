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
import numpy as np
import matplotlib.pyplot as plt
from one.api import ONE
from ibllib.atlas import BrainRegions
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import h5py
import sklearn
from sklearn.preprocessing import QuantileTransformer

# %% [markdown]
# p(BR|F) = p(F|BR)*p(BR)/p(F)

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
one = ONE(base_url="https://alyx.internationalbrainlab.org")
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

# %%
br = BrainRegions()

# %%
# coarse_region_feature_test = dict()
fine_region_feature = dict()

for i in range(len(Benchmark_pids)):
    pID =Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    
    #load brain region along the probe
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
    if channel_depths is not None:
        regions = channel_depths[regions]
    region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]
    
    #load z values
    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        z = h5['localizations'][:,2]
        max_ptps = h5['maxptps'][:]
        spike_times = h5["spike_index"][:,0]/30000
    
    which = slice(None);#max_ptps>6 #threshold on spikes' ptps
    
    #load waveform features
    ptp_durations = np.load(out_dir + '/ptp_duration.npy')
    pt_ratios = np.load(out_dir + '/peak_trough_ratio.npy')
    repolariztion_slope = np.load(out_dir + '/reploarization_slope_window_50.npy')
    recovery_slope = np.load(out_dir + '/recovery_slope.npy')
    spatial_spread = np.load(out_dir + '/non_threshold_spatial_spread.npy')
    
    #
    z = z[which]
    max_ptps = max_ptps[which]
    ptp_durations  = ptp_durations[which]
    pt_ratios = pt_ratios[which]
    repolariztion_slopes = repolariztion_slope[which]
    recovery_slopes  = recovery_slope[which]    
    spatial_spread  = spatial_spread[which]    
    spike_times = spike_times[which]
    #uniformly sample from BRs
    bins = np.unique(regions)
    inds = np.digitize(z, bins,right=True)
    
    inds = inds - 1 
    for j in range(len(region_labels)):
        which_region = np.where(inds == j)
        #split the recoding into two halves
        first_half_idx = np.where(spike_times[which_region]<90)
        second_half_idx = np.where(spike_times[which_region]>=90)
        # print(np.shape(first_half_idx)[1])
        # print(np.shape(second_half_idx)[1])
        # print((np.shape(first_half_idx)[1]>1000) & (np.shape(second_half_idx)[1]>1000))
        if (np.shape(first_half_idx)[1]>1000) & (np.shape(second_half_idx)[1]>1000):
            # print('ok')
            which_region = np.squeeze(np.where(inds == j))
            np.random.seed(0)
            subsample_1 = np.random.choice(which_region[first_half_idx], 1000, replace=False)
            np.random.seed(0)
            subsample_2 = np.random.choice(which_region[second_half_idx], 1000, replace=False)
            RL = region_labels[j,1]
            if RL in fine_region_feature.keys():
                fine_region_feature[RL]['max_ptps'] = np.append(fine_region_feature[RL]['max_ptps'], np.concatenate((max_ptps[subsample_1][:,None], max_ptps[subsample_2][:,None]), axis = 1), axis = 0)
                fine_region_feature[RL]['ptp_durations']= np.append(fine_region_feature[RL]['ptp_durations'], np.concatenate((ptp_durations[subsample_1][:,None], ptp_durations[subsample_2][:,None]), axis = 1)/300, axis = 0)
                fine_region_feature[RL]['pt_ratios']= np.append(fine_region_feature[RL]['pt_ratios'], np.concatenate((pt_ratios[subsample_1][:,None], pt_ratios[subsample_2][:,None]), axis = 1), axis = 0)
                fine_region_feature[RL]['repolariztion_slopes']= np.append(fine_region_feature[RL]['repolariztion_slopes'], np.concatenate((repolariztion_slopes[subsample_1][:,None], repolariztion_slopes[subsample_2][:,None]), axis = 1), axis = 0)
                fine_region_feature[RL]['recovery_slopes']= np.append(fine_region_feature[RL]['recovery_slopes'], np.concatenate((recovery_slopes[subsample_1][:,None], recovery_slopes[subsample_2][:,None]), axis = 1), axis = 0)
                fine_region_feature[RL]['spatial_spread']= np.append(fine_region_feature[RL]['spatial_spread'], np.concatenate((spatial_spread[subsample_1][:,None], spatial_spread[subsample_2][:,None]), axis = 1), axis = 0)
            else:
                fine_region_feature[RL] = dict()
                fine_region_feature[RL]['max_ptps'] = np.concatenate((max_ptps[subsample_1][:,None], max_ptps[subsample_2][:,None]), axis = 1)
                fine_region_feature[RL]['ptp_durations'] = np.concatenate((ptp_durations[subsample_1][:,None], ptp_durations[subsample_2][:,None]), axis = 1)/300
                fine_region_feature[RL]['pt_ratios'] = np.concatenate((pt_ratios[subsample_1][:,None], pt_ratios[subsample_2][:,None]), axis = 1)
                fine_region_feature[RL]['repolariztion_slopes'] = np.concatenate((repolariztion_slopes[subsample_1][:,None], repolariztion_slopes[subsample_2][:,None]), axis = 1)
                fine_region_feature[RL]['recovery_slopes'] = np.concatenate((recovery_slopes[subsample_1][:,None], recovery_slopes[subsample_2][:,None]), axis = 1)
                fine_region_feature[RL]['spatial_spread'] = np.concatenate((spatial_spread[subsample_1][:,None], spatial_spread[subsample_2][:,None]), axis = 1)


# %%
brain_regions = list(fine_region_feature.keys())
brain_region_prior = np.zeros(len(brain_regions))
for i in range(len(brain_regions)):
    brain_region_prior[i] = len(fine_region_feature[brain_regions[i]]['max_ptps'])

# %%
BR_interested_idx = np.where(brain_region_prior>1000)
BR_interested = np.array(brain_regions)[BR_interested_idx]

# %%
BR_interested

# %%
which_region = np.where(inds == 1)
spike_times[which_region]
np.shape(np.where(spike_times[which_region]>90))[1]

# %%
len(brain_regions)


# %%
def bandwidth_estimator(data, mod = True):
    n = len(data)
    assert n > 1
    std = np.std(data)
    iqr = scipy.stats.iqr(data)
    A = np.min([std, iqr/1.34])
    if mod:
        h = np.power(4/(3*n),1/5)*A
    else:
        h = 0.9*np.power(n,-1/5)*A
    return h


# %%
from matplotlib import cm

# %%
k = 3
bins = np.arange(feature_bins[k][0], feature_bins[k][1], feature_bins[k][2])
l = len(bins)
x = (bins[0:l-1] + bins[1:l])/2
# h = bandwidth_estimator(data, mod =False)
fig, axs = plt.subplots(4, 5, figsize = (40, 30))
h = np.array([0.1, 0.2, 0.5, 1])
colors = cm.winter(np.linspace(0,1,len(h)))
for i in range(20):
    data = fine_region_feature[brain_regions[i]][features[k]]

    # kernel1 = scipy.stats.gaussian_kde(data, bw_method =h)  #bandwidthestimator


    values, bins = np.histogram(data, bins = bins, density = True)
    r = i//5
    c = np.mod(i, 5)

    # plt.plot(x, kernel1(x))
    for j in range(len(h)):
        kernel= scipy.stats.gaussian_kde(data, bw_method =h[j])
        axs[r][c].plot(x, kernel(x), c = colors[j], label = h[j])

    axs[r][c].plot(x, values/values.sum()/feature_bins[k][2],c = 'grey', label = 'hist')
    axs[r][c].legend()


# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
feature_bins = [[6, 30, 0.5],[0.1, 0.8, 0.01], [-2.2, 2.2, 0.05], [5, 125, 1],[-10, 0, 0.1], [40, 100, 0.5]]

# %%
k = 2
feature = features[k]

bins = np.arange(feature_bins[k][0], feature_bins[k][1], feature_bins[k][2])
l = len(bins)
likelihood_maxptps = np.zeros([len(brain_regions), l-1])
all_ptps = []

for i in range(len(brain_regions)):
    values, bins = np.histogram(fine_region_feature[brain_regions[i]][feature], bins = bins, density = True)
    kernel = scipy.stats.gaussian_kde(fine_region_feature[brain_regions[i]][feature], bw_method = 0.2)
    x = (bins[0:l-1] + bins[1:l])/2
    # plt.plot(x, values)
    plt.plot(x, kernel(x))
    likelihood_maxptps[i, :] = values
    all_ptps = np.concatenate((all_ptps, fine_region_feature[brain_regions[i]][feature]))
    # plt.xlim([0, 40])

# %% jupyter={"outputs_hidden": true}
bins = np.arange(40, 100, 0.5)
l = len(bins)
likelihood_ptp_durations = np.zeros([len(brain_regions), l-1])
for i in range(len(brain_regions)):
    values, bins = np.histogram(fine_region_feature[brain_regions[i]]['spatial_spread'], bins = bins, density = False)
    x = (bins[0:l-1] + bins[1:l])/2
    plt.plot(x, values)
    likelihood_ptp_durations[i, :] = values
    # plt.xlim([0, 40])

# %% jupyter={"outputs_hidden": true}
plt.plot(x,likelihood_ptp_durations.sum(0)/likelihood_ptp_durations.sum().sum())

# %%
likelihood = likelihood_maxptps/np.repeat(likelihood_maxptps.sum(1)[:, None], l-1, axis = 1)

# %%
from sklearn.neighbors import KernelDensity
y = np.array(all_ptps)

kernel_sum = scipy.stats.gaussian_kde(y, bw_method = 0.1)
plt.plot(x, kernel_sum(x))


plt.plot(x, likelihood_maxptps.sum(0)/likelihood_maxptps.sum(0).sum()/feature_bins[k][2])

# %%
import sklearn
from sklearn.preprocessing import QuantileTransformer
fig, axs = plt.subplots(1,1, figsize = (6, 6))

inverse_cdf_data = sklearn.preprocessing.quantile_transform(y[:, None])
qt = QuantileTransformer()
qt.fit_transform(y[:, None])
transformed_data = qt.transform(fine_region_feature[brain_regions[i]][feature][:, None])
axs.scatter(y[:, None], inverse_cdf_data)
axs.scatter(fine_region_feature[brain_regions[i]][feature][:, None], transformed_data)
# axs[1].hist(transformed_data)

# %%
transform_data_all = qt.transform(all_data[:, None])
plt.hist(transform_data_all, bins = np.arange(0, 1, 0.1))

# %% jupyter={"outputs_hidden": true}
for k in range(6):
    feature = features[k]
    # bins = np.arange(feature_bins[k][0], feature_bins[k][1], feature_bins[k][2])
    bins = np.linspace(0, 1, 51)
    l = len(bins)
    x = (bins[0:l-1] + bins[1:l])/2
    # h = bandwidth_estimator(data, mod =False)
    fig, axs = plt.subplots(4, 5, figsize = (40, 30))
    h = np.array([0.1, 0.2, 0.5, 1])
    colors = cm.winter(np.linspace(0,1,len(h)))

    l = len(bins)
    likelihood = np.zeros([len(brain_regions), l-1])
    all_data = []

    for i in range(len(brain_regions)):
        data = fine_region_feature[brain_regions[i]][feature]
        if len(data)>1000:
            np.random.seed(0)
            sampled_idx[i,:] = np.random.randint(len(data), size = 1000)
        else:
            sampled_idx[i,:] = np.int32(np.arange(1000))
        all_data = np.concatenate((all_data, data[np.int32(sampled_idx[i,:]),0]))

    qt = QuantileTransformer()
    qt.fit_transform(all_data[:, None])

    for i in range(20):
        data = fine_region_feature[brain_regions[i]][feature][:,0]
        transformed_data = qt.transform(data[:, None])

        # kernel1 = scipy.stats.gaussian_kde(data, bw_method =h)  #bandwidthestimator


        values, bins = np.histogram(transformed_data, bins = bins, density = True)
        r = i//5
        c = np.mod(i, 5)

        # plt.plot(x, kernel1(x))
        for j in range(len(h)):
            kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =h[j])
            axs[r][c].plot(x, kernel(x), c = colors[j], label = h[j])

        axs[r][c].plot(x, values,c = 'grey', label = 'hist')
        axs[r][c].legend()
        axs[r][c].set_title(brain_regions[i], fontsize = 14)
    plt.savefig(feature + '_KDE_window_size_compare.png')

# %%
prior = brain_region_prior/brain_region_prior.sum()

# %%
norm = likelihood_maxptps.sum(0)/likelihood_maxptps.sum().sum()

# %%
posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)/ np.repeat(norm[:, None], len(brain_regions), axis = 1).T

# %%
plt.imshow(posterior)
plt.colorbar()

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
feature_bins = [[6, 30, 0.5],[0.1, 0.8, 0.01], [-2.2, 2.2, 0.05], [5, 125, 1],[-10, 0, 0.1], [40, 100, 0.5]]

# bw = [0.2, 0.1, 0.2, 0.1, 0.1, 0.1]

fig, axs = plt.subplots(1, 6, figsize = (60, 20))
for i in range(6):
    feature = features[i]
    bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
    l = len(bins)
    likelihood = np.zeros([len(brain_regions), l-1])
    for j in range(len(brain_regions)):
        values, bins = np.histogram(fine_region_feature[brain_regions[j]][feature], bins = bins)
        likelihood[j, :] = values
    norm = likelihood.sum(0)/likelihood.sum().sum()
    likelihood = likelihood/np.repeat(likelihood.sum(1)[:, None], l-1, axis = 1)
    prior = brain_region_prior/brain_region_prior.sum()
    posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)/ np.repeat(norm[:, None], len(brain_regions), axis = 1).T
    
    # v_low = np.percentile(posterior, 2.5)
    # v_high = np.percentile(posterior, 97.5)
    
    im = axs[i].imshow(posterior,aspect='auto', extent=[feature_bins[i][0], feature_bins[i][1], len(brain_regions), 0], vmin = 0, vmax = 0.05)
    # axs[i].colorbar()
    if i ==0:
        axs[i].set_ylabel('Brain regions', fontsize = 50)
    axs[i].set_xlabel(feature, fontsize = 50)
    axs[i].tick_params(labelsize=20)
    
    
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=20)
    

plt.suptitle('p(BR|F)', fontsize = 80)    

# %% jupyter={"outputs_hidden": true}
fine_region_feature[RL]

# %%
np.shape(np.concatenate([fine_region_feature[RL][x] for x in features], 1))

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
bw = 0.1
bins = np.linspace(0, 1, 51)
fig, axs = plt.subplots(1, 6, figsize = (60, 20))
for k in range(6):
    feature = features[k]
    # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
    l = len(bins)
    x = (bins[0:l-1] + bins[1:l])/2
    likelihood = np.zeros([len(brain_regions), l-1])
    
    all_data = []
    for i in range(len(brain_regions)):
        all_data = np.concatenate((all_data, fine_region_feature[brain_regions[i]][feature]))
    qt = QuantileTransformer()
    qt.fit_transform(all_data[:, None])
    
    for j in range(len(brain_regions)):
        data = fine_region_feature[brain_regions[j]][feature]
        transformed_data = qt.transform(data[:, None])
        kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
        likelihood[j,:] = kernel(x)/kernel(x).sum()

    values, bins = np.histogram(qt.transform(all_data[:, None]), bins = bins, density = True)
    norm = values/values.sum()
    prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
    # prior = np.ones(112,)*1/112
    posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)/np.repeat(norm[:, None], len(brain_regions), axis = 1).T
    
    # v_low = np.percentile(posterior, 2.5)
    # v_high = np.percentile(posterior, 97.5)
    
    im = axs[k].imshow(posterior,aspect='auto', extent=[0, 1, 108, 0], vmin = 0, vmax = 0.08)
    # axs[i].colorbar()
    if k ==0:
        axs[k].set_ylabel('Brain regions', fontsize = 50)
    axs[k].set_xlabel(feature, fontsize = 50)
    axs[k].tick_params(labelsize=20)
    
    
    divider = make_axes_locatable(axs[k])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=20)
    

plt.suptitle('p(BR|F) with inverse cdf and SDE', fontsize = 80)    

# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
feature_bins = [[6, 30, 0.5],[0.1, 0.8, 0.01], [-2.2, 2.2, 0.05], [5, 125, 1],[-10, 0, 0.1], [40, 100, 0.5]]
titles = ['first half', 'second half', 'combined']

# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
feature_bins = [[6, 30, 0.5],[0.1, 0.8, 0.01], [-2.2, 2.2, 0.05], [5, 125, 1],[-10, 0, 0.1], [40, 100, 0.5]]
titles = ['first half', 'second half', 'combined']
# bw = [0.2, 0.1, 0.2, 0.1, 0.1, 0.1]

grid = plt.GridSpec(3, 1)
fig, axs = plt.subplots(3, 6, figsize = (60, 60))
for n in range(3):
    title_holder = fig.add_subplot(grid[n])
    title_holder.set_title(titles[n], fontsize = 50)
    title_holder.set_axis_off()
    for i in range(6):
        feature = features[i]
        bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
        l = len(bins)
        likelihood = np.zeros([len(brain_regions), l-1])

        for j in range(len(brain_regions)):
            if (n == 0) | (n == 1):
                data = fine_region_feature[brain_regions[j]][feature][:,n]
            else:
                data = fine_region_feature[brain_regions[j]][feature][:,:]
                
            if len(data)>pick_n:
                np.random.seed(0)
                sampled_idx = np.random.randint(len(data), size = pick_n)
                if n == 2:
                    data = data[sampled_idx,:].flatten()
                else:
                    data = data[sampled_idx]
                
            values, bins = np.histogram(data, bins = bins)
            likelihood[j, :] = values
            
        norm = likelihood.sum(0)/likelihood.sum().sum()
        likelihood = likelihood/np.repeat(likelihood.sum(1)[:, None], l-1, axis = 1)
        prior = np.ones(len(brain_regions),)*1/len(brain_regions)
        posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)/ np.repeat(norm[:, None], len(brain_regions), axis = 1).T

        # v_low = np.percentile(posterior, 2.5)
        # v_high = np.percentile(posterior, 97.5)

        im = axs[n][i].imshow(posterior,aspect='auto', extent=[feature_bins[i][0], feature_bins[i][1], 112, 0], vmin = 0, vmax = 0.05)
        # axs[i].colorbar()
        if i ==0:
            axs[n][i].set_ylabel('Brain regions', fontsize = 50)
        axs[n][i].set_xlabel(feature, fontsize = 50)
        axs[n][i].tick_params(labelsize=20)


        divider = make_axes_locatable(axs[n][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.tick_params(labelsize=20)
    

plt.suptitle('p(BR|F)', fontsize = 80) 
plt.savefig('p(BR|F)_binned_count.png')

# %% [markdown]
# overlaped regions only:

# %% jupyter={"outputs_hidden": true}
from mpl_toolkits.axes_grid1 import make_axes_locatable
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
bw = 0.1
bins = np.linspace(0, 1, 100)
grid = plt.GridSpec(3, 1)
fig, axs = plt.subplots(3, 6, figsize = (60, 60))
sort_idx_all = np.zeros((6, len(BR_interested)))
for n in range(3):
    title_holder = fig.add_subplot(grid[n])
    title_holder.set_title(titles[n], fontsize = 50)
    title_holder.set_axis_off()
    for k in range(6):
        feature = features[k]
        # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
        l = len(bins)
        x = (bins[0:l-1] + bins[1:l])/2
        likelihood = np.zeros([len(BR_interested), l-1])

        test_all_data = []
        train_all_data = []
        train_data = np.zeros((len(BR_interested),2000))
        test_data = np.zeros((len(BR_interested),2000))
        
        for i in range(len(BR_interested)):
            data =  fine_region_feature[BR_interested[i]][feature]
            data_length = len(data)
            num_datasets = len(data)//1000
            if len(data)>2000:
                if np.mod(num_datasets,2):
                    np.random.seed(0)
                    sampled_train_idx = np.random.randint((num_datasets//2+1)*1000, size = 1000)
                    np.random.seed(0)
                    sampled_test_idx = np.random.randint(num_datasets//2*1000, size = 1000)
                    train_data[i,:] = data[sampled_train_idx,:].flatten()
                    test_data[i,:] = data[sampled_test_idx + (num_datasets//2+1)*1000,:].flatten()
                else:
                    np.random.seed(0)
                    sampled_idx = np.random.randint(num_datasets//2*1000, size = 1000)
                    train_data[i,:] = data[sampled_idx,:].flatten()
                    test_data[i,:] = data[sampled_idx + (num_datasets//2)*1000,:].flatten()
            else:
                sampled_idx = np.arange(1000)
                train_data[i,:] = data[sampled_idx,:].flatten()
                test_data[i,:] = data[sampled_idx + (num_datasets//2)*1000,:].flatten()

            test_all_data = np.concatenate((test_all_data, test_data[i,:]))
            train_all_data = np.concatenate((train_all_data, train_data[i,:]))
        
        assert len(test_all_data) == 2000*len(BR_interested)
        assert len(train_all_data) == 2000*len(BR_interested)
        
        qt = QuantileTransformer()
        # if n == 0:
        #     qt.fit_transform(train_all_data[:, None])
        # elif n == 1:
        #     qt.fit_transform(test_all_data[:, None])
        # else:
        qt.fit_transform(np.concatenate((train_all_data[:, None], test_all_data[:, None]), axis = 1).flatten()[:, None])                             
                                            
                                    
        for j in range(len(BR_interested)):
            if n == 0:
                data = train_data[j,:][:, None]
            elif n == 1:
                data = test_data[j,:][:, None]
            else:
                data = np.concatenate((train_data[j,:][:, None], test_data[j,:][:, None]), axis = 1).flatten()[:, None]
            
            transformed_data = qt.transform(data)
            kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
            likelihood[j,:] = kernel(x)/kernel(x).sum()

        values, bins = np.histogram(qt.transform(all_data[:, None]), bins = bins, density = True)
        norm = values/values.sum()
        # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
        prior = np.ones(len(BR_interested),)*1/len(BR_interested)
        posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], 112, axis = 1).T
        # posterior = posterior/np.linalg.norm(posterior)
        # inverse_value = inverse_transform(x)
        if n == 0:
            max_idx = np.argmax(posterior, axis = 1)
            sort_idx_all[k, :] = np.argsort(max_idx)
        sort_idx = np.int32(sort_idx_all[k, :])

        im = axs[n][k].imshow(posterior[sort_idx,:],aspect='auto', extent=[0, 1, 16, 0], vmin = 0, vmax = 0.12)
        # axs[i].colorbar()
        if i ==0:
            axs[n][k].set_ylabel('Brain regions', fontsize = 50)
        axs[n][k].set_xlabel(feature, fontsize = 50)
        axs[n][k].tick_params(labelsize=20)
        axs[n][k].set_yticks(np.arange(17))
        axs[n][k].set_yticklabels(BR_interested[sort_idx], fontsize = 20)


        divider = make_axes_locatable(axs[n][k])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.set_yticks([0,0.1])
        cax.tick_params(labelsize=20)


plt.suptitle('p(BR|F)', fontsize = 80)
# plt.savefig('p(BR|F)_with_inverse_cdf_and_KDE.png')

# %% [markdown]
# seperate plots:

# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
bw = 0.1
bins = np.linspace(0, 1, 100)
grid = plt.GridSpec(3, 1)

sort_idx_all = np.zeros((6, len(BR_interested)))
for n in range(3):
    fig, axs = plt.subplots(1, 6, figsize = (60, 20))
    title_holder = fig.add_subplot(grid[0])
    title_holder.set_title(titles[n], fontsize = 50)
    title_holder.set_axis_off()
    for k in range(6):
        feature = features[k]
        # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
        l = len(bins)
        x = (bins[0:l-1] + bins[1:l])/2
        likelihood = np.zeros([len(BR_interested), l-1])

        test_all_data = []
        train_all_data = []
        train_data = np.zeros((len(BR_interested),2000))
        test_data = np.zeros((len(BR_interested),2000))
        
        for i in range(len(BR_interested)):
            data =  fine_region_feature[BR_interested[i]][feature]
            data_length = len(data)
            num_datasets = len(data)//1000
            if len(data)>2000:
                if np.mod(num_datasets,2):
                    np.random.seed(0)
                    sampled_train_idx = np.random.randint((num_datasets//2+1)*1000, size = 1000)
                    np.random.seed(0)
                    sampled_test_idx = np.random.randint(num_datasets//2*1000, size = 1000)
                    train_data[i,:] = data[sampled_train_idx,:].flatten()
                    test_data[i,:] = data[sampled_test_idx + (num_datasets//2+1)*1000,:].flatten()
                else:
                    np.random.seed(0)
                    sampled_idx = np.random.randint(num_datasets//2*1000, size = 1000)
                    train_data[i,:] = data[sampled_idx,:].flatten()
                    test_data[i,:] = data[sampled_idx + (num_datasets//2)*1000,:].flatten()
            else:
                sampled_idx = np.arange(1000)
                train_data[i,:] = data[sampled_idx,:].flatten()
                test_data[i,:] = data[sampled_idx + (num_datasets//2)*1000,:].flatten()

            test_all_data = np.concatenate((test_all_data, test_data[i,:]))
            train_all_data = np.concatenate((train_all_data, train_data[i,:]))
        
        assert len(test_all_data) == 2000*len(BR_interested)
        assert len(train_all_data) == 2000*len(BR_interested)
        
        qt = QuantileTransformer()
        # if n == 0:
        #     qt.fit_transform(train_all_data[:, None])
        # elif n == 1:
        #     qt.fit_transform(test_all_data[:, None])
        # else:
        qt.fit_transform(np.concatenate((train_all_data[:, None], test_all_data[:, None]), axis = 1).flatten()[:, None])                             
                                            
                                    
        for j in range(len(BR_interested)):
            if n == 0:
                data = train_data[j,:][:, None]
            elif n == 1:
                data = test_data[j,:][:, None]
            else:
                data = np.concatenate((train_data[j,:][:, None], test_data[j,:][:, None]), axis = 1).flatten()[:, None]
            
            transformed_data = qt.transform(data)
            kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
            likelihood[j,:] = kernel(x)/kernel(x).sum()

        values, bins = np.histogram(qt.transform(all_data[:, None]), bins = bins, density = True)
        norm = values/values.sum()
        # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
        prior = np.ones(len(BR_interested),)*1/len(BR_interested)
        posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], 112, axis = 1).T
        # posterior = posterior/np.linalg.norm(posterior)
        # inverse_value = inverse_transform(x)
        if n == 0:
            max_idx = np.argmax(posterior, axis = 1)
            sort_idx_all[k, :] = np.argsort(max_idx)
        sort_idx = np.int32(sort_idx_all[k, :])

        im = axs[k].imshow(posterior[sort_idx,:],aspect='auto', extent=[0, 1, 16, 0], vmin = 0, vmax = 0.12)
        # axs[i].colorbar()
        if i ==0:
            axs[k].set_ylabel('Brain regions', fontsize = 50)
        axs[k].set_xlabel(feature, fontsize = 50)
        axs[k].tick_params(labelsize=20)
        axs[k].set_yticks(np.arange(17))
        axs[k].set_yticklabels(BR_interested[sort_idx], fontsize = 20)


        divider = make_axes_locatable(axs[k])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.set_yticks([0,0.1])
        cax.tick_params(labelsize=20)


    plt.suptitle('p(BR|F)', fontsize = 80)
    plt.savefig('p(BR|F)_'+titles[n]+'.png')

# %%
confusion matrix (univariate & mutivariate)
error vs n
simple linear classifier
transformers


# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
bw = 0.1
bins = np.linspace(0, 1, 51)
grid = plt.GridSpec(3, 1)
fig, axs = plt.subplots(3, 6, figsize = (60, 60))
for n in range(3):
    title_holder = fig.add_subplot(grid[n])
    title_holder.set_title(titles[n], fontsize = 50)
    title_holder.set_axis_off()
    for k in range(6):
        feature = features[k]
        # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
        l = len(bins)
        x = (bins[0:l-1] + bins[1:l])/2
        likelihood = np.zeros([len(brain_regions), l-1])

        all_data = []
        sampled_idx = np.zeros((len(brain_regions), 1000))
        for i in range(len(brain_regions)):
            data =  fine_region_feature[brain_regions[i]][feature]
            if len(data)>1000:
                np.random.seed(0)
                sampled_idx[i,:] = np.random.randint(len(data), size = 1000)
            else:
                sampled_idx[i,:] = np.int32(np.arange(1000))
     
            if n == 0 | n == 1:
                all_data = np.concatenate((all_data, data[np.int32(sampled_idx[i,:]),n]))
            else:
                all_data = np.concatenate((all_data, data[np.int32(sampled_idx[i,:]),:].flatten()))
        
        if n==0 | n==1:
            assert len(all_data) == 1000*len(brain_regions)
        else:
            assert len(all_data) == 2000*len(brain_regions)
        
        qt = QuantileTransformer()
        qt.fit_transform(all_data[:, None])
    
        for j in range(len(brain_regions)):
            if n == 0| n == 1:
                data = fine_region_feature[brain_regions[j]][feature][np.int32(sampled_idx[j,:]),n]
            else:
                data = fine_region_feature[brain_regions[j]][feature][np.int32(sampled_idx[j,:]),:].flatten()
            
            transformed_data = qt.transform(data[:, None])
            kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
            likelihood[j,:] = kernel(x)/kernel(x).sum()

        values, bins = np.histogram(qt.transform(all_data[:, None]), bins = bins, density = True)
        norm = values/values.sum()
        # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
        prior = np.ones(len(brain_regions),)*1/len(brain_regions)
        posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], 112, axis = 1).T
        # posterior = posterior/np.linalg.norm(posterior)
        # inverse_value = inverse_transform(x)

        im = axs[n][k].imshow(posterior,aspect='auto', extent=[0, 1, 112, 0], vmin = 0, vmax = 0.03)
        # axs[i].colorbar()
        if i ==0:
            axs[n][k].set_ylabel('Brain regions', fontsize = 50)
        axs[n][k].set_xlabel(feature, fontsize = 50)
        axs[n][k].tick_params(labelsize=20)


        divider = make_axes_locatable(axs[n][k])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.tick_params(labelsize=20)


plt.suptitle('p(BR|F) with inverse cdf and KDE', fontsize = 80)
# plt.savefig('p(BR|F)_with_inverse_cdf_and_KDE.png')

# %% [markdown]
# - convert to cosmos label
# - compare distribution of finer parcellation and coarser parcellation
# - compute new posterior

# %%
allen_label = np.unique(df_channels['acronym'])
cosmos_label = br.remap(br.acronym2id(allen_label), source_map='Allen', target_map='Cosmos')

# %% jupyter={"outputs_hidden": true}
br.remap(br.acronym2id(allen_label), source_map='Allen', target_map='Cosmos')

# %%
np.unique(cosmos_label, return_index = True)

# %%
cosmos_region_labels = br.remap(br.acronym2id(region_labels[:,1]), source_map='Allen', target_map='Cosmos')
regions[cosmos_region_labels == 703][0]


# %%
def map_z_to_cosmos(allen_labels, allen_z):
    #input: allen labeling 
    allen_acronym = allen_labels
    cosmos_region_labels = br.remap(br.acronym2id(allen_acronym), source_map='Allen', target_map='Cosmos')
    
    unique_cosmos = np.unique(cosmos_region_labels)
    
    allen_acronym = []
    for i in range(len(unique_cosmos)):
        allen_acronym.append(unique_cosmos[i])
        cosmos_regions = allen_z[cosmos_region_labels == unique_cosmos[i]]
        combined_cosmos_regions = cosmos_regions[0][None,:]
        for j in range(1, len(cosmos_regions)):
            if cosmos_regions[j,0] == combined_cosmos_regions[-1, 1]:
                combined_cosmos_regions[-1, 1] = cosmos_regions[j,1]
            else:
                combined_cosmos_regions = np.concatenate((combined_cosmos_regions, cosmos_regions[j][None,:]), axis = 0)
                allen_acronym.append(unique_cosmos[i])
    return allen_acronym, combined_cosmos_regions


# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']


df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
allen_label = np.unique(df_channels['acronym'])
cosmos_label = br.remap(br.acronym2id(allen_label), source_map='Allen', target_map='Cosmos')
cosmos_label = np.unique(cosmos_label)

coarse_region_feature = dict()

for i in cosmos_label:
    coarse_region_feature[i] = dict()
    for j in features:
        coarse_region_feature[i][j] = None
    
for i in range(len(Benchmark_pids)):
    pID =Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    
    #load brain region along the probe
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)
    
    channel_ids = df_channels['atlas_id'].values

    region_info = br.get(channel_ids)
    boundaries = np.where(np.diff(region_info.id) != 0)[0]
    boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]

    regions = np.c_[boundaries[0:-1], boundaries[1:]]
    
    channel_depths=df_channels['axial_um'].values
    if channel_depths is not None:
        regions = channel_depths[regions]
    region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]
    
    
    cosmos_region_labels, cosmos_regions = map_z_to_cosmos(region_labels[:,1], regions)
    #######
    
    #load z values
    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        z = h5['localizations'][:,2]
        max_ptps = h5['maxptps'][:]
        spike_times = h5["spike_index"][:,0]/30000
    
    which = slice(None);#max_ptps>6 #threshold on spikes' ptps
    
    #load waveform features
    ptp_durations = np.load(out_dir + '/ptp_duration.npy')
    pt_ratios = np.load(out_dir + '/peak_trough_ratio.npy')
    repolariztion_slope = np.load(out_dir + '/reploarization_slope_window_50.npy')
    recovery_slope = np.load(out_dir + '/recovery_slope.npy')
    spatial_spread = np.load(out_dir + '/non_threshold_spatial_spread.npy')
    
    #
    z = z[which]
    max_ptps = max_ptps[which]
    ptp_durations  = ptp_durations[which]
    pt_ratios = pt_ratios[which]
    repolariztion_slopes = repolariztion_slope[which]
    recovery_slopes  = recovery_slope[which]    
    spatial_spread  = spatial_spread[which]    
    spike_times = spike_times[which]
    #uniformly sample from BRs
    bins = np.unique(cosmos_regions)
    inds = np.digitize(z, bins,right=True)
    
    inds = inds - 1 
    
    
    for j in range(len(cosmos_region_labels)):
        which_region = np.where(inds == j)
        #split the recoding into two halves
        first_half_idx = np.where(spike_times[which_region]<90)
        second_half_idx = np.where(spike_times[which_region]>=90)
        # print(np.shape(first_half_idx)[1])
        # print(np.shape(second_half_idx)[1])
        # print((np.shape(first_half_idx)[1]>1000) & (np.shape(second_half_idx)[1]>1000))
        if (np.shape(first_half_idx)[1]>1000) & (np.shape(second_half_idx)[1]>1000):
            # print('ok')
            which_region = np.squeeze(np.where(inds == j))
            RL = cosmos_region_labels[j]
            
            try:  
                coarse_region_feature[RL]['max_ptps'] = np.append(coarse_region_feature[RL]['max_ptps'], max_ptps[which_region][:,None], axis = 0)
                coarse_region_feature[RL]['ptp_durations']= np.append(coarse_region_feature[RL]['ptp_durations'], ptp_durations[which_region][:,None]/300, axis = 0)
                coarse_region_feature[RL]['pt_ratios']= np.append(coarse_region_feature[RL]['pt_ratios'], pt_ratios[which_region][:,None], axis = 0)
                coarse_region_feature[RL]['repolariztion_slopes']= np.append(coarse_region_feature[RL]['repolariztion_slopes'], repolariztion_slopes[which_region][:,None], axis = 0)
                coarse_region_feature[RL]['recovery_slopes']= np.append(coarse_region_feature[RL]['recovery_slopes'], recovery_slopes[which_region][:,None], axis = 0)
                coarse_region_feature[RL]['spatial_spread']= np.append(coarse_region_feature[RL]['spatial_spread'], spatial_spread[which_region][:,None], axis = 0)
            except:
                coarse_region_feature[RL]['max_ptps'] = max_ptps[which_region][:,None]
                coarse_region_feature[RL]['ptp_durations']= ptp_durations[which_region][:,None]/300
                coarse_region_feature[RL]['pt_ratios']= pt_ratios[which_region][:,None]
                coarse_region_feature[RL]['repolariztion_slopes']= repolariztion_slopes[which_region][:,None]
                coarse_region_feature[RL]['recovery_slopes']= recovery_slopes[which_region][:,None]
                coarse_region_feature[RL]['spatial_spread']= spatial_spread[which_region][:,None]

# %%
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

df_channels = df_channels.reset_index(drop=False)

# %%
for i in range(len(cosmos_label)):
    RL = cosmos_label[i]
    if coarse_region_feature[RL]['spatial_spread'] is None:
        print(str(RL)+': no value')
    else:
        print(str(RL)+ ':' + str(len(coarse_region_feature[RL]['spatial_spread'])))

# %%
cosmos_regions_list = [313, 315, 512, 549, 623, 698, 703, 997, 1065]
fig, axs = plt.subplots(1, 6, figsize = (60, 10))

for j in range(len(features)):
    F = features[j]
    higher_bound = np.percentile(np.concatenate([coarse_region_feature[x][F] for x in cosmos_regions_list], 0), 97.5)
    lower_bound = np.percentile(np.concatenate([coarse_region_feature[x][F] for x in cosmos_regions_list], 0), 2.5)
    for i in range(len(cosmos_regions_list)):
        r = i//5
        c = np.mod(i, 5)
        RL = cosmos_regions_list[i]
        y = coarse_region_feature[RL][F].T
        x = np.linspace(lower_bound, higher_bound, 51)
        kernel_sum = scipy.stats.gaussian_kde(y, bw_method = 0.1)
        axs[j].plot(x, kernel_sum(x))
        axs[j].set_title(F)
        axs[j].legend(br.id2acronym(cosmos_regions_list))
        axs[j].set_ylabel('probability density')
        axs[j].set_xlabel(F)
plt.savefig('coarse_parcellation_distribution_all_spikes.png')

# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']


df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
allen_label = np.unique(df_channels['acronym'])
cosmos_label = br.remap(br.acronym2id(allen_label), source_map='Allen', target_map='Cosmos')
cosmos_label = np.unique(cosmos_label)

# fig, axs = plt.subplots(len(Benchmark_pids), 6, figsize = (20, 50))
        
###
# split into test and training
all_idx = np.arange(len(Benchmark_pids))
for test_idx in all_idx:
    train_idx = np.delete(all_idx, test_idx)

    ###
    coarse_region_feature_train = dict()

    for i in cosmos_label:
        coarse_region_feature_train[i] = dict()
        for j in features:
            coarse_region_feature_train[i][j] = None        

    for i in train_idx:
        pID =Benchmark_pids[i]
        eID, probe = one.pid2eid(pID)
        out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

        #load brain region along the probe
        df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

        df_channels = df_channels.reset_index(drop=False)
        df_channels = df_channels[df_channels.pid == pID]
        df_channels = df_channels.reset_index(drop=True)

        channel_ids = df_channels['atlas_id'].values

        region_info = br.get(channel_ids)
        boundaries = np.where(np.diff(region_info.id) != 0)[0]
        boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]

        regions = np.c_[boundaries[0:-1], boundaries[1:]]

        channel_depths=df_channels['axial_um'].values
        if channel_depths is not None:
            regions = channel_depths[regions]
        region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]


        cosmos_region_labels, cosmos_regions = map_z_to_cosmos(region_labels[:,1], regions)
        #######

        #load z values
        h5_path = out_dir + '/' + 'subtraction.h5'
        with h5py.File(h5_path) as h5:
            z = h5['localizations'][:,2]
            max_ptps = h5['maxptps'][:]
            spike_times = h5["spike_index"][:,0]/30000

        which = slice(None);#max_ptps>6 #threshold on spikes' ptps

        #load waveform features
        ptp_durations = np.load(out_dir + '/ptp_duration.npy')
        pt_ratios = np.load(out_dir + '/peak_trough_ratio.npy')
        repolariztion_slope = np.load(out_dir + '/reploarization_slope_window_50.npy')
        recovery_slope = np.load(out_dir + '/recovery_slope.npy')
        spatial_spread = np.load(out_dir + '/non_threshold_spatial_spread.npy')

        #
        z = z[which]
        max_ptps = max_ptps[which]
        ptp_durations  = ptp_durations[which]
        pt_ratios = pt_ratios[which]
        repolariztion_slopes = repolariztion_slope[which]
        recovery_slopes  = recovery_slope[which]    
        spatial_spread  = spatial_spread[which]    
        spike_times = spike_times[which]
        #uniformly sample from BRs
        bins = np.unique(cosmos_regions)
        inds = np.digitize(z, bins,right=True)

        inds = inds - 1 


        for j in range(len(cosmos_region_labels)):
            which_region = np.where(inds == j)

            RL = cosmos_region_labels[j]
            
            if len(which_region) == 0:
                continue

            try:  
                coarse_region_feature_train[RL]['max_ptps'] = np.append(coarse_region_feature_train[RL]['max_ptps'], max_ptps[which_region][:,None], axis = 0)
                coarse_region_feature_train[RL]['ptp_durations']= np.append(coarse_region_feature_train[RL]['ptp_durations'], ptp_durations[which_region][:,None]/300, axis = 0)
                coarse_region_feature_train[RL]['pt_ratios']= np.append(coarse_region_feature_train[RL]['pt_ratios'], pt_ratios[which_region][:,None], axis = 0)
                coarse_region_feature_train[RL]['repolariztion_slopes']= np.append(coarse_region_feature_train[RL]['repolariztion_slopes'], repolariztion_slopes[which_region][:,None], axis = 0)
                coarse_region_feature_train[RL]['recovery_slopes']= np.append(coarse_region_feature_train[RL]['recovery_slopes'], recovery_slopes[which_region][:,None], axis = 0)
                coarse_region_feature_train[RL]['spatial_spread']= np.append(coarse_region_feature_train[RL]['spatial_spread'], spatial_spread[which_region][:,None], axis = 0)
            except:
                coarse_region_feature_train[RL]['max_ptps'] = max_ptps[which_region][:,None]
                coarse_region_feature_train[RL]['ptp_durations']= ptp_durations[which_region][:,None]/300
                coarse_region_feature_train[RL]['pt_ratios']= pt_ratios[which_region][:,None]
                coarse_region_feature_train[RL]['repolariztion_slopes']= repolariztion_slopes[which_region][:,None]
                coarse_region_feature_train[RL]['recovery_slopes']= recovery_slopes[which_region][:,None]
                coarse_region_feature_train[RL]['spatial_spread']= spatial_spread[which_region][:,None]          
    ################       
    # get test data
    coarse_region_feature_test = dict()

    for i in cosmos_label:
        coarse_region_feature_test[i] = dict()
        for j in features:
            coarse_region_feature_test[i][j] = None 


    pID =Benchmark_pids[test_idx]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

    #load brain region along the probe
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)

    channel_ids = df_channels['atlas_id'].values

    region_info = br.get(channel_ids)
    boundaries = np.where(np.diff(region_info.id) != 0)[0]
    boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]

    regions = np.c_[boundaries[0:-1], boundaries[1:]]

    channel_depths=df_channels['axial_um'].values
    if channel_depths is not None:
        regions = channel_depths[regions]
    region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]


    cosmos_region_labels, cosmos_regions = map_z_to_cosmos(region_labels[:,1], regions)
    #######

    #load z values
    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        z = h5['localizations'][:,2]
        max_ptps = h5['maxptps'][:]
        spike_times = h5["spike_index"][:,0]/30000

    which = slice(None);#max_ptps>6 #threshold on spikes' ptps

    #load waveform features
    ptp_durations = np.load(out_dir + '/ptp_duration.npy')
    pt_ratios = np.load(out_dir + '/peak_trough_ratio.npy')
    repolariztion_slope = np.load(out_dir + '/reploarization_slope_window_50.npy')
    recovery_slope = np.load(out_dir + '/recovery_slope.npy')
    spatial_spread = np.load(out_dir + '/non_threshold_spatial_spread.npy')

    #
    z = z[which]
    max_ptps = max_ptps[which]
    ptp_durations  = ptp_durations[which]
    pt_ratios = pt_ratios[which]
    repolariztion_slopes = repolariztion_slope[which]
    recovery_slopes  = recovery_slope[which]    
    spatial_spread  = spatial_spread[which]    
    spike_times = spike_times[which]
    #uniformly sample from BRs
    bins = np.unique(cosmos_regions)
    inds = np.digitize(z, bins,right=True)

    inds = inds - 1 


    for j in range(len(cosmos_region_labels)):
        which_region = np.where(inds == j)

        which_region = np.squeeze(np.where(inds == j))
        RL = cosmos_region_labels[j]
        
        if len(which_region) == 0:
            continue

        try:  
            coarse_region_feature_test[RL]['max_ptps'] = np.append(coarse_region_feature_test[RL]['max_ptps'], max_ptps[which_region][:,None], axis = 0)
            coarse_region_feature_test[RL]['ptp_durations']= np.append(coarse_region_feature_test[RL]['ptp_durations'], ptp_durations[which_region][:,None]/300, axis = 0)
            coarse_region_feature_test[RL]['pt_ratios']= np.append(coarse_region_feature_test[RL]['pt_ratios'], pt_ratios[which_region][:,None], axis = 0)
            coarse_region_feature_test[RL]['repolariztion_slopes']= np.append(coarse_region_feature_test[RL]['repolariztion_slopes'], repolariztion_slopes[which_region][:,None], axis = 0)
            coarse_region_feature_test[RL]['recovery_slopes']= np.append(coarse_region_feature_test[RL]['recovery_slopes'], recovery_slopes[which_region][:,None], axis = 0)
            coarse_region_feature_test[RL]['spatial_spread']= np.append(coarse_region_feature_test[RL]['spatial_spread'], spatial_spread[which_region][:,None], axis = 0)
        except:
            coarse_region_feature_test[RL]['max_ptps'] = max_ptps[which_region][:,None]
            coarse_region_feature_test[RL]['ptp_durations']= ptp_durations[which_region][:,None]/300
            coarse_region_feature_test[RL]['pt_ratios']= pt_ratios[which_region][:,None]
            coarse_region_feature_test[RL]['repolariztion_slopes']= repolariztion_slopes[which_region][:,None]
            coarse_region_feature_test[RL]['recovery_slopes']= recovery_slopes[which_region][:,None]
            coarse_region_feature_test[RL]['spatial_spread']= spatial_spread[which_region][:,None]
    ######################################
    
    uniformly_sampled_coarse_region_feature_train = coarse_region_feature_train.copy()
    
    RL_in_training = []
    
    for RL in cosmos_regions_list:
        RL_spike_n = len(uniformly_sampled_coarse_region_feature_train[RL]['max_ptps'])
        if RL_spike_n>0:
            RL_in_training.append(RL)
            idx_pick = np.arange(RL_spike_n)
            np.random.seed(40)
            idx_pick = np.random.choice(idx_pick, 200000, replace=True)
            for F in features:
                uniformly_sampled_coarse_region_feature_train[RL][F] = uniformly_sampled_coarse_region_feature_train[RL][F][idx_pick]

    ######################################
    posterior_list = dict()
    qt_transformer_list = dict()

    bins = np.linspace(0, 1, 101)
    
    for k in range(6):
        feature = features[k]
        # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
        l = len(bins)
        x = (bins[0:l-1] + bins[1:l])/2
        likelihood = np.zeros([len(BR_interested), l-1])

        train_all_data = [uniformly_sampled_coarse_region_feature_train[x][feature] for x in cosmos_regions_list] 
        train_all_data = np.concatenate(train_all_data, 0)

        qt_transformer_list[feature] = QuantileTransformer()
        qt_transformer_list[feature].fit_transform(train_all_data)

        for j in range(len(RL_in_training)):
        # for j in range(len(BR_interested)):
            data = uniformly_sampled_coarse_region_feature_train[RL_in_training[j]][feature]

            transformed_data = qt_transformer_list[feature].transform(data)
            kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
            likelihood[j,:] = kernel(x)/kernel(x).sum()

        values, bins = np.histogram(qt.transform(all_data[:, None]), bins = bins, density = True)
        norm = values/values.sum()
        # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
        prior = np.ones(len(BR_interested),)*1/len(BR_interested)
        posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], 112, axis = 1).T
        posterior_list[feature] = posterior
        
    
    #####################################
    real_region = []
    pred_region = []
    x = np.linspace(0, 1, 101)
    bins = np.arange(0, 1.0001, 1/100)
    fig, axs = plt.subplots(1, 6, figsize = (60, 10))
    for i in range(len(features)):
        F = features[i]
        posterior = posterior_list[F]
        qt = qt_transformer_list[F]
        for j in range(len(BR_interested)):
            RL = BR_interested[j]
            spike_features = coarse_region_feature_test[RL][F]
            if spike_features is None:
                continue
            else:
                if RL in RL_in_training:
                    transformed_spk_feature = qt.transform(spike_features)
                    spk_n = len(transformed_spk_feature)
                    pred_idx = np.digitize(transformed_spk_feature, bins)
                    corrected_pred_idx = np.copy(pred_idx)
                    corrected_pred_idx[pred_idx == 101] = 100

                    pred_cosmos = np.array(BR_interested)[np.squeeze(np.argmax(posterior[:,corrected_pred_idx-1], axis = 0))]

                    real_region = np.concatenate((real_region, np.ones(spk_n) * RL))
                    pred_region = np.concatenate((pred_region, pred_cosmos))
        axs[i].imshow(confusion_matrix(real_region, pred_region, normalize = 'true', labels=BR_interested))
        axs[i].set_title(F)
        axs[i].set_xlabel('pred')
        axs[i].set_ylabel('true')
        axs[i].set_xticks(np.arange(len(BR_interested)) + 0.5)
        axs[i].set_yticks(np.arange(len(BR_interested)) + 0.5)
        
        axs[i].set_xticklabels(list(br.id2acronym(BR_interested)))
        axs[i].set_yticklabels(list(br.id2acronym(BR_interested)))
        fig.suptitle(Benchmark_pids[test_idx])

# %%
1

# %%
for i in range(len(cosmos_label)):
    RL = cosmos_label[i]
    if coarse_region_feature_test[RL]['spatial_spread'] is None:
        print(str(RL)+': no value')
    else:
        print(str(RL)+ ':' + str(len(coarse_region_feature_test[RL]['spatial_spread'])))

# %%
for RL in cosmos_regions_list:
    if coarse_region_feature_train[RL]['spatial_spread'] is None:
        print(str(RL)+': no value')
    else:
        print(str(RL)+ ':' + str(len(coarse_region_feature_train[RL]['spatial_spread'])))

# %%
for RL in cosmos_regions_list:
    if uniformly_sampled_coarse_region_feature_train[RL]['spatial_spread'] is None:
        print(str(RL)+': no value')
    else:
        print(str(RL)+ ':' + str(len(uniformly_sampled_coarse_region_feature_train[RL]['spatial_spread'])))

# %% jupyter={"outputs_hidden": true}
coarse_region_feature_train

# %%
uniformly_sampled_coarse_region_feature_train = coarse_region_feature_train.copy()
for RL in cosmos_regions_list:
    RL_spike_n = len(uniformly_sampled_coarse_region_feature_train[RL]['max_ptps'])
    idx_pick = np.arange(RL_spike_n)
    np.random.seed(40)
    idx_pick = np.random.choice(idx_pick, 200000, replace=True)
    for F in features:
        uniformly_sampled_coarse_region_feature_train[RL][F] = uniformly_sampled_coarse_region_feature_train[RL][F][idx_pick]

# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
bw = 0.1
bins = np.linspace(0, 1, 101)
grid = plt.GridSpec(3, 1)

BR_interested_acronym = np.array(br.id2acronym(cosmos_regions_list))
BR_interested = cosmos_regions_list

sort_idx_all = np.zeros((6, len(BR_interested)))
fig, axs = plt.subplots(1, 6, figsize = (60, 20))
title_holder = fig.add_subplot(grid[0])
title_holder.set_title(titles[n], fontsize = 50)
title_holder.set_axis_off()
n = 0

posterior_list = dict()
qt_transformer_list = dict()
for k in range(6):
    feature = features[k]
    # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
    l = len(bins)
    x = (bins[0:l-1] + bins[1:l])/2
    likelihood = np.zeros([len(BR_interested), l-1])

    train_all_data = [uniformly_sampled_coarse_region_feature_train[x][feature] for x in cosmos_regions_list] 
    train_all_data = np.concatenate(train_all_data, 0)

    qt_transformer_list[feature] = QuantileTransformer()
    qt_transformer_list[feature].fit_transform(train_all_data)
    



    for j in range(len(BR_interested)):
        data = uniformly_sampled_coarse_region_feature_train[BR_interested[j]][feature]

        transformed_data = qt_transformer_list[feature].transform(data)
        kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
        likelihood[j,:] = kernel(x)/kernel(x).sum()

    values, bins = np.histogram(qt.transform(all_data[:, None]), bins = bins, density = True)
    norm = values/values.sum()
    # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
    prior = np.ones(len(BR_interested),)*1/len(BR_interested)
    posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], 112, axis = 1).T
    posterior_list[feature] = posterior
    # posterior = posterior/np.linalg.norm(posterior)
    # inverse_value = inverse_transform(x)
    if n == 0:
        max_idx = np.argmax(posterior, axis = 1)
        sort_idx_all[k, :] = np.argsort(max_idx)
    sort_idx = np.int32(sort_idx_all[k, :])

    im = axs[k].imshow(posterior[sort_idx,:],aspect='auto', extent=[0, 1, 9, 0], vmin = 0, vmax = 0.2)
    # axs[i].colorbar()
    if i ==0:
        axs[k].set_ylabel('Brain regions', fontsize = 50)
    axs[k].set_xlabel(feature, fontsize = 50)
    axs[k].tick_params(labelsize=20)
    axs[k].set_yticks(np.arange(len(BR_interested_acronym))+0.5)
    axs[k].set_yticklabels(BR_interested_acronym[sort_idx], fontsize = 20)


    divider = make_axes_locatable(axs[k])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.set_yticks([0,0.1])
    cax.tick_params(labelsize=20)


# plt.suptitle('p(BR|F)', fontsize = 80)
# plt.savefig('p(BR|F)_'+titles[n]+'.png')

# %% jupyter={"outputs_hidden": true}
from sklearn.metrics import confusion_matrix
real_region = []
pred_region = []
x = np.linspace(0, 1, 101)
bins = np.arange(0, 1.0001, 1/100)
fig, axs = plt.subplots(1, 6, figsize = (60, 10))
for i in range(len(features)):
    F = features[i]
    posterior = posterior_list[F]
    qt = qt_transformer_list[F]
    for j in range(len(BR_interested)):
        RL = BR_interested[j]
        spike_features = coarse_region_feature_test[RL][F]
        if spike_features is None:
            continue
        else:
            transformed_spk_feature = qt.transform(spike_features)
            spk_n = len(transformed_spk_feature)
            pred_idx = np.digitize(transformed_spk_feature, bins)
            corrected_pred_idx = np.copy(pred_idx)
            corrected_pred_idx[pred_idx == 101] = 100
            
            pred_cosmos = np.array(BR_interested)[np.squeeze(np.argmax(posterior[:,corrected_pred_idx-1], axis = 0))]
            
            real_region = np.concatenate((real_region, np.ones(spk_n) * RL))
            pred_region = np.concatenate((pred_region, pred_cosmos))
    axs[i].imshow(confusion_matrix(real_region, pred_region))
    axs[i].set_title(F)

# %%
cm = confusion_matrix(real_region, pred_region)

# %% jupyter={"outputs_hidden": true}
from sklearn.metrics import confusion_matrix
plt.imshow(confusion_matrix(real_region, pred_region))

# %%
pred_idx = np.digitize(transformed_spk_feature, bins)
corrected_pred_idx = np.copy(pred_idx)
corrected_pred_idx[pred_idx == 101] = 100
plt.hist(corrected_pred_idx, bins = np.arange(100))

# %%
qt_transformer_list['max_ptps']

# %%
for key, value in coarse_region_feature_test.iteritems():
    temp = [key,value]
    dictlist.append(temp)

# %%

# %%
