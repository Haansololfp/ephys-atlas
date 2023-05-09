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
from sklearn.metrics import confusion_matrix
from brainbox.ephys_plots import plot_brain_regions

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
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
br = BrainRegions()


# %%
def map_z_to_cosmos(allen_labels, allen_z):
    #input: allen labeling 
    allen_acronym = allen_labels
    cosmos_region_labels = br.remap(br.acronym2id(allen_acronym), source_map='Allen', target_map='Cosmos')
    unique_cosmos, uniq_idx = np.unique(cosmos_region_labels, return_index=True)
    unique_cosmos = unique_cosmos[np.argsort(uniq_idx)]
    cosmos_labels = []
    n = 0
    for i in range(len(unique_cosmos)):
        cosmos_labels.append(unique_cosmos[i])
        cosmos_regions = allen_z[cosmos_region_labels == unique_cosmos[i]]
        # combined_cosmos_regions = np.concatenate((combined_cosmos_regions, cosmos_regions[0][None,:]), axis = 0)
        for j in range(len(cosmos_regions)):
            n = n+1
            if n == 1:
                combined_cosmos_regions = allen_z[j][None,:]
            else:
                if j == 0:
                    combined_cosmos_regions = np.concatenate((combined_cosmos_regions, cosmos_regions[j][None,:]), axis = 0)
                else:
                    if cosmos_regions[j,0] <= combined_cosmos_regions[-1, 1]:
                        combined_cosmos_regions[-1, 1] = cosmos_regions[j,1]
                    else:
                        combined_cosmos_regions = np.concatenate((combined_cosmos_regions, cosmos_regions[j][None,:]), axis = 0)
                        cosmos_labels.append(unique_cosmos[i])
    return cosmos_labels, combined_cosmos_regions


# %%
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)


allen_label = np.unique(df_channels['acronym'])
cosmos_label = br.remap(br.acronym2id(allen_label), source_map='Allen', target_map='Cosmos')

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
    coarse_region_feature[i]['nPID'] = 0
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
        # first_half_idx = np.where(spike_times[which_region]<90)
        # second_half_idx = np.where(spike_times[which_region]>=90)
        # # print(np.shape(first_half_idx)[1])
        # # print(np.shape(second_half_idx)[1])
        # # print((np.shape(first_half_idx)[1]>1000) & (np.shape(second_half_idx)[1]>1000))
        # if (np.shape(first_half_idx)[1]>1000) & (np.shape(second_half_idx)[1]>1000):
            # print('ok')
        RL = cosmos_region_labels[j]
        if len(which_region)!=0:
            coarse_region_feature[RL]['nPID'] += 1
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
[coarse_region_feature[RL]['nPID'] for RL in cosmos_regions_list]

# %%
br.id2acronym(cosmos_regions_list[4])

# %%
uniformly_sampled_coarse_region_feature = coarse_region_feature.copy()
    
RL_in_training = []

for RL in cosmos_regions_list:
    RL_spike_n = len(uniformly_sampled_coarse_region_feature[RL]['max_ptps'])
    if RL_spike_n>0:
        RL_in_training.append(RL)
        idx_pick = np.arange(RL_spike_n)
        np.random.seed(40)
        idx_pick = np.random.choice(idx_pick, 200000, replace=True)
        for F in features:
            uniformly_sampled_coarse_region_feature[RL][F] = uniformly_sampled_coarse_region_feature[RL][F][idx_pick]


# %%
posterior_list = dict()
qt_transformer_list = dict()

bins = np.linspace(0, 1, 101)

for k in range(6):
    feature = features[k]
    # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
    l = len(bins)
    x = (bins[0:l-1] + bins[1:l])/2
    likelihood = np.zeros([len(BR_interested), l-1])

    train_all_data = [uniformly_sampled_coarse_region_feature[x][feature] for x in cosmos_regions_list] 
    train_all_data = np.concatenate(train_all_data, 0)

    qt_transformer_list[feature] = QuantileTransformer()
    qt_transformer_list[feature].fit_transform(train_all_data)

    for j in range(len(RL_in_training)):
    # for j in range(len(BR_interested)):
        data = uniformly_sampled_coarse_region_feature[RL_in_training[j]][feature]

        transformed_data = qt_transformer_list[feature].transform(data)
        kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
        likelihood[j,:] = kernel(x)/kernel(x).sum()

    values, bins = np.histogram(qt_transformer_list[feature].transform(train_all_data), bins = bins, density = True)
    norm = values/values.sum()
    # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
    prior = np.ones(len(BR_interested),)*1/len(BR_interested)
    posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], 112, axis = 1).T
    posterior_list[feature] = posterior

# %%
cosmos_regions_list = [313, 315, 512, 549, 623, 698, 703, 997, 1065]
fig, axs = plt.subplots(1, 6, figsize = (60, 10))

for j in range(len(features)):
    F = features[j]
    higher_bound = np.percentile(np.concatenate([uniformly_sampled_coarse_region_feature[x][F] for x in cosmos_regions_list], 0), 99.5)
    lower_bound = np.percentile(np.concatenate([uniformly_sampled_coarse_region_feature[x][F] for x in cosmos_regions_list], 0), 0.5)
    for i in range(len(cosmos_regions_list)):
        r = i//5
        c = np.mod(i, 5)
        RL = cosmos_regions_list[i]
        y = uniformly_sampled_coarse_region_feature[RL][F].T
        x = np.linspace(lower_bound, higher_bound, 51)
        kernel_sum = scipy.stats.gaussian_kde(y, bw_method = 0.1)
        axs[j].plot(x, kernel_sum(x), c = colors[i])
        axs[j].set_title(F)
        axs[j].legend(br.id2acronym(cosmos_regions_list))
        axs[j].set_ylabel('probability density')
        axs[j].set_xlabel(F)
plt.savefig('coarse_parcellation_distribution_all_spikes.png')

# %%
# c = Colormap('tab10', len(BR_interested))
n = len(BR_interested)
colors = plt.cm.tab10(np.linspace(0,1,n))

# %%
colors

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
bw = 0.1
bins = np.linspace(0, 1, 101)
grid = plt.GridSpec(3, 1)

BR_interested_acronym = np.array(br.id2acronym(cosmos_regions_list))
BR_interested = cosmos_regions_list

# sort_idx_all = np.zeros((6, len(BR_interested)))
fig, axs = plt.subplots(1, 6, figsize = (60, 20))
# title_holder = fig.add_subplot(grid[0])
# title_holder.set_title(titles[n], fontsize = 50)
# title_holder.set_axis_off()
# n = 0

posterior_list = dict()
qt_transformer_list = dict()
for k in range(6):
    feature = features[k]
    # bins = np.arange(feature_bins[i][0], feature_bins[i][1], feature_bins[i][2])
    l = len(bins)
    x = (bins[0:l-1] + bins[1:l])/2
    likelihood = np.zeros([len(BR_interested), l-1])

    train_all_data = [uniformly_sampled_coarse_region_feature[x][feature] for x in cosmos_regions_list] 
    train_all_data = np.concatenate(train_all_data, 0)

    qt_transformer_list[feature] = QuantileTransformer()
    qt_transformer_list[feature].fit_transform(train_all_data)
    



    for j in range(len(BR_interested)):
        data = uniformly_sampled_coarse_region_feature[BR_interested[j]][feature]

        transformed_data = qt_transformer_list[feature].transform(data)
        kernel= scipy.stats.gaussian_kde(transformed_data.T, bw_method =bw)
        likelihood[j,:] = kernel(x)/kernel(x).sum()

    values, bins = np.histogram(qt_transformer_list[feature].transform(train_all_data), bins = bins, density = True)
    norm = values/values.sum()
    # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
    prior = np.ones(len(BR_interested),)*1/len(BR_interested)
    posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], 112, axis = 1).T
    posterior_list[feature] = posterior
    # posterior = posterior/np.linalg.norm(posterior)
    # inverse_value = inverse_transform(x)
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


plt.suptitle('p(BR|F)', fontsize = 80)
plt.savefig('p(BR|F)_coarse_parcellation'+'.png')

# %% jupyter={"outputs_hidden": true}
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

        values, bins = np.histogram(qt_transformer_list[feature].transform(train_all_data), bins = bins, density = True)
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
        # fig.suptitle(Benchmark_pids[test_idx])

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'

# %%
cosmos_color

# %%
features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']
cosmos_regions_list = [313, 315, 512, 549, 623, 698, 703, 997, 1065]
BR_interested = cosmos_regions_list


df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
allen_label = np.unique(df_channels['acronym'])
cosmos_label = br.remap(br.acronym2id(allen_label), source_map='Allen', target_map='Cosmos')
cosmos_label = np.unique(cosmos_label)

BR_interested_idx = np.concatenate([np.where(br.id==i) for i in BR_interested])
cosmos_color = br.rgb[BR_interested_idx]
# fig, axs = plt.subplots(len(Benchmark_pids), 6, figsize = (20, 50))
cosmos_color[7] = [0, 0, 0]   
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

        values, bins = np.histogram(qt_transformer_list[feature].transform(train_all_data), bins = bins, density = True)
        norm = values/values.sum()
        # prior = np.squeeze(brain_region_prior/brain_region_prior.sum())
        prior = np.ones(len(BR_interested),)*1/len(BR_interested)
        posterior = likelihood * np.repeat(prior[:, None], l-1,axis = 1)*len(x)#/np.repeat(norm[:, None], len(BR_interested), axis = 1).T
        posterior_list[feature] = posterior/posterior.sum(axis=0,keepdims=1)

    #####################################

    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        x = h5["localizations"][:,0]
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


    ##############
    #pick the right spikes
    which = (z > regions[0][0]) & (z < regions[-1][1])

    features = ['max_ptps', 'ptp_durations', 'pt_ratios', 'repolariztion_slopes', 'recovery_slopes', 'spatial_spread']

    F_test = dict()

    x = x[which]
    z = z[which]


    real_cosmos_region = np.nan*np.ones(len(x))

    for i in range(len(cosmos_region_labels)):
        idx = np.where(BR_interested == cosmos_region_labels[i])
        if np.shape(idx)[1]!=0:
            spk_in_BR = np.where((z <= cosmos_regions[i, 1]) & (z > cosmos_regions[i, 0]))
            real_cosmos_region[spk_in_BR] = idx
            
    discard_region_idx = np.where(~np.isnan(real_cosmos_region))
    
    x = x[discard_region_idx]
    z = z[discard_region_idx]
    F_test[features[0]] = max_ptps[which][discard_region_idx]
    F_test[features[1]] = ptp_durations[which][discard_region_idx]
    F_test[features[2]] = pt_ratios[which][discard_region_idx]
    F_test[features[3]] = repolariztion_slope[which][discard_region_idx]
    F_test[features[4]] = recovery_slope[which][discard_region_idx]
    F_test[features[5]] = spatial_spread[which][discard_region_idx]
    real_cosmos_region = real_cosmos_region[discard_region_idx]
    
    spike_n = len(x)


    ###
    pred_cosmos_posterior = np.ones((len(BR_interested),spike_n))
    ###
    for i in range(len(features)):
            F = features[i]
            posterior = posterior_list[F]
            qt = qt_transformer_list[F]
            data = F_test[F]
            transformed_spk_feature = qt.transform(data[:,None])
            pred_idx = np.digitize(transformed_spk_feature, bins)
            corrected_pred_idx = np.copy(pred_idx)
            corrected_pred_idx[pred_idx == 101] = 100

            pred_cosmos_posterior = pred_cosmos_posterior * np.squeeze(posterior[:,corrected_pred_idx-1])


    pred_cosmos_idx = np.argmax(pred_cosmos_posterior, axis = 0)

    pred_cosmos_certainty = scipy.stats.entropy(np.ones(len(BR_interested))/len(BR_interested)) - scipy.stats.entropy(pred_cosmos_posterior)

    pred_cosmos_probability = pred_cosmos_posterior[np.int32(real_cosmos_region), np.arange(len(x))]
    
    pred_cosmos_max_probability = np.max(pred_cosmos_posterior, axis = 0)
    
    fig, axs = plt.subplots(1, 2, figsize = (8, 60))
    
    alpha = 0.25 + 0.74 * (pred_cosmos_max_probability - np.min(pred_cosmos_max_probability))/(np.max(pred_cosmos_max_probability) - np.min(pred_cosmos_max_probability))
    # alpha = (pred_cosmos_probability - np.min(pred_cosmos_probability))/(np.max(pred_cosmos_probability) - np.min(pred_cosmos_probability))

    # axs[0].scatter(x, z, c = cosmos_color[pred_cosmos_idx]/255, s = pred_cosmos_certainty*10, alpha = alpha)
    axs[0].scatter(x, z, c = colors[pred_cosmos_idx, 0:3], s = pred_cosmos_certainty*20, alpha = alpha)
    axs[0].set_ylim([20, 3840])



    plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Cosmos'), channel_depths=df_channels['axial_um'].values,
                               brain_regions=br, display=True, ax=axs[1], label='right')
    
    plt.suptitle(pID)
    plt.savefig('posterior_estimate_' + pID + '.png')

# %%
br_change_color = br

# %%
for i in range(len(BR_interested)):
    BL = BR_interested[i]
    br_change_color.rgb[br_change_color.id == BL] = np.int32(colors[i][None, 0:3]*255)

# %%
br_change_color

# %%
np.shape(np.max(pred_cosmos_posterior, axis = 0))

# %%
np.where(np.isnan(real_cosmos_region))

# %%
len(data)

# %%
alpha

# %%
map_z_to_cosmos(region_labels[:,1], regions)

# %%
regions

# %%
region_labels[:,1]

# %% jupyter={"outputs_hidden": true}
allen_labels = region_labels[:,1]
allen_z = regions


allen_acronym = allen_labels
cosmos_region_labels = br.remap(br.acronym2id(allen_acronym), source_map='Allen', target_map='Cosmos')
print(cosmos_region_labels)
unique_cosmos, uniq_idx = np.unique(cosmos_region_labels, return_index=True)
unique_cosmos = unique_cosmos[np.argsort(uniq_idx)]
cosmos_labels = []
n = 0
for i in range(len(unique_cosmos)):
    cosmos_labels.append(unique_cosmos[i])
    cosmos_regions = allen_z[cosmos_region_labels == unique_cosmos[i]]
    print(cosmos_regions)
    # combined_cosmos_regions = np.concatenate((combined_cosmos_regions, cosmos_regions[0][None,:]), axis = 0)
    for j in range(len(cosmos_regions)):
        n = n+1
        if n == 1:
            combined_cosmos_regions = allen_z[j][None,:]
        else:
            if j == 0:
                combined_cosmos_regions = np.concatenate((combined_cosmos_regions, cosmos_regions[j][None,:]), axis = 0)
            else:
                if cosmos_regions[j,0] <= combined_cosmos_regions[-1, 1]:
                    combined_cosmos_regions[-1, 1] = cosmos_regions[j,1]
                    print('combine:')
                    print(combined_cosmos_regions)
                else:
                    combined_cosmos_regions = np.concatenate((combined_cosmos_regions, cosmos_regions[j][None,:]), axis = 0)
                    cosmos_labels.append(unique_cosmos[i])
                    print('seperate:')
                    print(combined_cosmos_regions)

# %%
combined_cosmos_regions

# %%
cosmos_labels

# %%
cosmos_region_labels

# %%
cosmos_region_labels, cosmos_regions = map_z_to_cosmos(region_labels[:,1], regions)

# %%
br.remap(br.acronym2id(region_labels[:,1]), source_map='Allen', target_map='Cosmos')

# %%
regions

# %%
(scipy.stats.entropy(np.ones(9)/9) - scipy.stats.entropy(posterior))*5

# %%
br.rgb(BR_interested)

# %%
br = BrainRegions()

# %%
br.rgb

# %%
br.id

# %%
BR_interested

# %%
np.concatenate([np.where(br.id==i) for i in BR_interested])

# %%
br.rgb[np.concatenate([np.where(br.id==i) for i in BR_interested])]

# %%
