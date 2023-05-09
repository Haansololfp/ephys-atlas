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
import matplotlib as mpl
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
from scipy import stats
from scipy.ndimage import gaussian_filter

# %%
from one.api import ONE
one = ONE(base_url="https://alyx.internationalbrainlab.org")

# %%
from ibllib.atlas import BrainRegions
from brainbox.ephys_plots import plot_brain_regions
import scipy
import pandas as pd

# %%
from pathlib import Path
from one.remote import aws
from one.api import ONE

LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/clusters.pqt'))
df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/probes.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/depths.pqt'))
df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/raw_ephys_features.pqt'))




# %%
df_voltage.info()

# %%
plt.imshow(np.tile(df_voltage['psd_alpha']['1a276285-8b0e-4cc9-9f0a-a3a002978724'][:,None],(1, 100)))

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
peak_value_bin_means

# %%
with h5py.File(h5_path) as h5:
    geom = h5['geom'][:]


# %%
def nan_gaussian_filt(U):
    sigma=2.0                  # standard deviation for Gaussian kernel
    truncate=4.0               # truncate filter at this many sigmas

    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma,truncate=truncate)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=sigma,truncate=truncate)

    Z=VV/WW
    
    return Z


# %%
np.shape(nan_gaussian_filt(ret.statistic))

# %%
heatmap = non_threshold_spatial_bin_means.T
higher_th = np.percentile(heatmap[:],95)
lower_th = np.percentile(heatmap[:],5)
heatmap = np.clip(heatmap, lower_th, higher_th)
plt.imshow(heatmap, aspect='auto')

# %%
import seaborn as sns

# %% jupyter={"outputs_hidden": true}
# main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets'
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'

bin_size = 40
z_reg_bins = np.arange(20, 3840, bin_size)
x_bins = [-60, 100]#np.arange(-16, 6, 16)
# z_reg_bins = np.arange(320, 3500, bin_size)
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    # out_dir = main_dir + '/' + eID + '_' + probe
    h5_path = out_dir + '/' + 'subtraction.h5'    
    try:
        h5 = h5py.File(h5_path)

        x = h5["localizations"][:,0]
        z_reg = h5["localizations"][:,2]
        maxptps = h5["maxptps"][:]
    except:
        continue
    
    h5.close()
    print(i)
    # try:
        # peak_value = np.load(out_dir + '/peak_value.npy')
    ptp_duration = np.load(out_dir + '/ptp_duration.npy')
    # halfpeak_duration = np.load(out_dir + '/halfpeak_duration.npy')
    peak_trough_ratio = np.load(out_dir + '/peak_trough_ratio.npy')

    spatial_spread = np.load(out_dir + '/spatial_spread_th12.npy')
    reploarization_slope = np.load(out_dir + '/reploarization_slope_window_50.npy')
    # reploarization_slope = np.load(out_dir + '/reploarization_slope.npy')
    recovery_slope = np.load(out_dir + '/recovery_slope.npy')
    spatial_non_threshold = np.load(out_dir + '/non_threshold_spatial_spread.npy')


        
    #     spatial_spread = np.load(out_dir + '/spatial_spread.npy')
    #     velocity = np.load(out_dir + '/velocity_th_25.npy')
    #     ci_err = np.load(out_dir + '/velocity_ci_th_25.npy')
    # except:
    #     continue
    

    # hdb_results = np.load('/moto/stats/users/hy2562/projects/monkey_insertion/pacman-task_c_220121_neu_insertion2_g0_imec0/pointsource_hdbscan.npz')
    # spike_train = hdb_results["arr_0"]
    # which = spike_train[:,1]>0

    which = maxptps>6#slice(None)

    fig = plt.figure(constrained_layout=True,figsize=(48,40))
    subplots = fig.subfigures(1,2, width_ratios=[3, 2])

    ax_a, ax_b = subplots[0].subfigures(2,1)
    axs1 = ax_a.subplots(1,9)
    axs2 = ax_b.subplots(1,9)
    axs = subplots[1].subplots(6,1)

    cmps = np.clip(spatial_non_threshold[which], 60, 90)
    
    
    ret = stats.binned_statistic_2d(x[which], z_reg[which], spatial_non_threshold[which], statistic='mean', bins=[x_bins, z_reg_bins])
    non_threshold_spatial_bin_means = nan_gaussian_filt(ret.statistic)
    # non_threshold_spatial_bin_means = ret.statistic
    
    
    nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
    im1 = axs1[3].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
    axs1[3].set_title('spatial \n spread',fontsize = 30)
    axs1[3].set_ylim([20 , 3840])
    axs1[3].set_xlim([-60, 100])
    axs1[3].set_yticklabels([])
    
    axs1_divider = make_axes_locatable(axs1[3])
    caxs1 = axs1_divider.append_axes("right", size="7%", pad="2%")
    cbs1 = fig.colorbar(im1, cax=caxs1)
    # axs[0].set_yticklabels([])
    
    ###########
    heatmap = non_threshold_spatial_bin_means.T
    higher_th = np.percentile(heatmap[:],95)
    lower_th = np.percentile(heatmap[:],5)
    heatmap = np.clip(heatmap, lower_th, higher_th)
    
    m_im1 = axs2[3].imshow(non_threshold_spatial_bin_means.T, aspect='auto')
    
    m_axs1_divider = make_axes_locatable(axs2[3])
    m_caxs1 = m_axs1_divider.append_axes("right", size="7%", pad="2%")
    m_cbs1 = fig.colorbar(m_im1, cax=m_caxs1)
    
    ###


    cmps = np.clip(reploarization_slope[which], 10, 100)
    
    ret = stats.binned_statistic_2d(x[which], z_reg[which], reploarization_slope[which], statistic='mean', bins=[x_bins, z_reg_bins])
    reploarization_slope_bin_means = nan_gaussian_filt(ret.statistic)
    # reploarization_slope_bin_means = ret.statistic
    
    nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
    im2 = axs1[4].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
    axs1[4].set_title('reploarization \n slope',fontsize = 30)
    axs1[4].set_ylim([20 , 3840])
    axs1[4].set_xlim([-60, 100])
    axs1[4].set_yticklabels([])
    
    axs2_divider = make_axes_locatable(axs1[4])
    caxs2 = axs2_divider.append_axes("right", size="7%", pad="2%")
    cbs2 = fig.colorbar(im2, cax=caxs2)
    
    #############
    heatmap = reploarization_slope_bin_means.T
    higher_th = np.percentile(heatmap[:],95)
    lower_th = np.percentile(heatmap[:],5)
    heatmap = np.clip(heatmap, lower_th, higher_th)
    
    m_im2 = axs2[4].imshow(heatmap, aspect='auto')
    
    m_axs2_divider = make_axes_locatable(axs2[4])
    m_caxs2 = m_axs2_divider.append_axes("right", size="7%", pad="2%")
    m_cbs2 = fig.colorbar(m_im2, cax=m_caxs2)
    
    ###
    
    cmps = np.clip(ptp_duration[which]/300, 0.2, 0.6)
    
    ret = stats.binned_statistic_2d(x[which],z_reg[which], ptp_duration[which]/300, statistic='mean', bins=[x_bins, z_reg_bins])
    ptp_duration_bin_means = nan_gaussian_filt(ret.statistic)
    # ptp_duration_bin_means = ret.statistic
    
    nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
    im3 = axs1[2].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
    axs1[2].set_title('ptp \n duration',fontsize = 30)
    axs1[2].set_ylim([20 , 3840])
    axs1[2].set_xlim([-60, 100])
    axs1[2].set_yticklabels([])

    axs3_divider = make_axes_locatable(axs1[2])
    caxs3 = axs3_divider.append_axes("right", size="7%", pad="2%")
    cbs3 = fig.colorbar(im3, cax=caxs3)
        
    #############
    heatmap = ptp_duration_bin_means.T
    higher_th = np.percentile(heatmap[:],95)
    lower_th = np.percentile(heatmap[:],5)
    heatmap = np.clip(heatmap, lower_th, higher_th)
    
    m_im3 = axs2[2].imshow(heatmap, aspect='auto')
    
    m_axs3_divider = make_axes_locatable(axs2[2])
    m_caxs3 = m_axs3_divider.append_axes("right", size="7%", pad="2%")
    m_cbs3 = fig.colorbar(m_im3, cax=m_caxs3)
    
    ###

    cmps = np.clip(peak_trough_ratio[which], -1.2, 1.2)
    
    ret = stats.binned_statistic_2d(x[which],z_reg[which], peak_trough_ratio[which], statistic='mean', bins=[x_bins, z_reg_bins])
    peak_trough_ratio_bin_means = nan_gaussian_filt(ret.statistic)
    # peak_trough_ratio_bin_means = ret.statistic
    
    nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
    im4 = axs1[0].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['RdBu'], rasterized=True)
    axs1[0].set_title('peak_trough \n ratio',fontsize = 30)
    axs1[0].set_ylim([20 , 3840])
    axs1[0].set_xlim([-60, 100])
    # axs1[0].set_yticklabels([])
    
    axs4_divider = make_axes_locatable(axs1[0])
    caxs4 = axs4_divider.append_axes("right", size="7%", pad="2%")
    cbs4 = fig.colorbar(im4, cax=caxs4)
    
    #############
    # if i  == 2:
    #     m_im4 = axs2[0].imshow(peak_trough_ratio_bin_means.T, aspect='auto', vmin=-0.8, vmax=0.8, cmap=mpl.colormaps['RdBu'], extent=[0,1,3840,20])
    # else:
    #     m_im4 = axs2[0].imshow(peak_trough_ratio_bin_means.T, aspect='auto', vmin = -1.2, vmax = -0.8, cmap=mpl.colormaps['RdBu'], extent=[0,1,3840,20])
    heatmap = peak_trough_ratio_bin_means.T
    higher_th = np.percentile(heatmap[:],95)
    lower_th = np.percentile(heatmap[:],5)
    heatmap = np.clip(heatmap, lower_th, higher_th)
    
    m_im4 = axs2[0].imshow(heatmap, aspect='auto', cmap=mpl.colormaps['RdBu'], extent=[0,1,3840,20])
   
    m_axs4_divider = make_axes_locatable(axs2[0])
    m_caxs4 = m_axs4_divider.append_axes("right", size="7%", pad="2%")
    m_cbs4 = fig.colorbar(m_im4, cax=m_caxs4)
    
    ###
    
#     cmps = np.clip(spatial_spread[which], 10, 100)
    
#     ret = stats.binned_statistic_2d(x[which],z_reg[which], spatial_spread[which], statistic='mean', bins=[x_bins, z_reg_bins])
#     # spatial_spread_bin_means = np.squeeze(gaussian_filter(ret.statistic, sigma=1))
#     spatial_spread_bin_means = ret.statistic
    
#     nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
#     im5 = axs1[3].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
#     axs1[3].set_title('spatial_spread')
#     axs1[3].set_ylim([20 , 3840])
#     axs1[3].set_xlim([-60, 100])
#     axs1[3].set_yticklabels([])
    
#     axs5_divider = make_axes_locatable(axs1[3])
#     caxs5 = axs5_divider.append_axes("right", size="7%", pad="2%")
#     cbs5 = fig.colorbar(im5, cax=caxs5)
    
#     #############
    
#     m_im5 = axs2[3].imshow(spatial_spread_bin_means.T, aspect='auto', vmin=30, vmax=80)
    
#     m_axs5_divider = make_axes_locatable(axs2[3])
#     m_caxs5 = m_axs5_divider.append_axes("right", size="7%", pad="2%")
#     m_cbs5 = fig.colorbar(m_im5, cax=m_caxs5)
    
    ###

#     ab_v = velocity[which]
#     notnan_idx = ~np.isnan(velocity[which])
#     # ab_v[np.isnan(velocity[which])] = 0
#     ab_v[np.abs(ab_v) - ci_err[which] < 0] = 0

#     cmps = np.clip(ab_v[notnan_idx], 0, 2)
#     nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
#     # nmps = np.clip(nmps, 0.01, 1)
#     im6 = axs[5].scatter(x[which][notnan_idx], z_reg[which][notnan_idx], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
#     axs[5].set_title('1/v_above')
#     axs[5].set_ylim([0 , 3820])
#     axs[5].set_xlim([-60, 100])
#     axs[5].set_yticklabels([])
    
#     ax6_divider = make_axes_locatable(axs[5])
#     cax6 = ax6_divider.append_axes("right", size="7%", pad="2%")
#     cb6 = fig.colorbar(im6, cax=cax6)
    
     ###
    
    # cmps = np.clip(maxptps[which], 8, 14)
    cmps = np.clip(maxptps[which], 6, 14)
    
    ret = stats.binned_statistic_2d(x[which], z_reg[which], maxptps[which], statistic='mean', bins=[x_bins, z_reg_bins])
    maxptps_bin_means = nan_gaussian_filt(ret.statistic)
    # maxptps_bin_means = ret.statistic
    
    nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
    im6 = axs1[1].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
    axs1[1].set_title('maxptps',fontsize = 30)
    axs1[1].set_ylim([20 , 3840])
    axs1[1].set_xlim([-60, 100])
    axs1[1].set_yticklabels([])
    
    axs6_divider = make_axes_locatable(axs1[1])
    caxs6 = axs6_divider.append_axes("right", size="7%", pad="2%")
    cbs6 = fig.colorbar(im6, cax=caxs6)
    
    #############
    heatmap = maxptps_bin_means.T
    higher_th = np.percentile(heatmap[:],95)
    lower_th = np.percentile(heatmap[:],5)
    heatmap = np.clip(heatmap, lower_th, higher_th)
    
    m_im6 = axs2[1].imshow(heatmap, aspect='auto')
    
    m_axs6_divider = make_axes_locatable(axs2[1])
    m_caxs6 = m_axs6_divider.append_axes("right", size="7%", pad="2%")
    m_cbs6 = fig.colorbar(m_im6, cax=m_caxs6)    
    
    ###
    
    cmps = np.clip(recovery_slope[which], -10, -1)
    
    ret = stats.binned_statistic_2d(x[which], z_reg[which], recovery_slope[which], statistic='mean', bins=[x_bins, z_reg_bins])
    recovery_slope_means = nan_gaussian_filt(ret.statistic)
    # recovery_slope_means = ret.statistic
    
    nmps = (cmps - cmps.min()) / (cmps.max() - cmps.min())
    im7 = axs1[5].scatter(x[which], z_reg[which], marker = '.', c=cmps, alpha=nmps, s=0.1, cmap=mpl.colormaps['viridis'], rasterized=True)
    axs1[5].set_title('recovery \n slope',fontsize = 30)
    axs1[5].set_ylim([20 , 3840])
    axs1[5].set_xlim([-60, 100])
    axs1[5].set_yticklabels([])
    
    axs7_divider = make_axes_locatable(axs1[5])
    caxs7 = axs7_divider.append_axes("right", size="7%", pad="2%")
    cbs7 = fig.colorbar(im7, cax=caxs7)
    
    #############
    heatmap = recovery_slope_means.T
    higher_th = np.percentile(heatmap[:],95)
    lower_th = np.percentile(heatmap[:],5)
    heatmap = np.clip(heatmap, lower_th, higher_th)
    
    m_im7 = axs2[5].imshow(heatmap, aspect='auto')
    # axs2[6].set_title('binned average recovery_slope')
    
    m_axs7_divider = make_axes_locatable(axs2[5])
    m_caxs7 = m_axs7_divider.append_axes("right", size="7%", pad="2%")
    m_cbs7 = fig.colorbar(m_im7, cax=m_caxs7)      
    ###
    
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
    

    br = BrainRegions()


    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)


    
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs1[6], title='Real', label='right')
    
    
    ######
    
    psd_theta = df_voltage['psd_theta'][pID].to_numpy()
    psd_gamma = df_voltage['psd_gamma'][pID].to_numpy()
    
    im_lfp1 = axs2[6].imshow(np.repeat(np.tile(psd_theta[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
    axs2[6].set_title('psd_theta',fontsize = 30)
    # axs3[1].set_ylim([32 , 350])
    # axs3[1].set_xlim([0, 100])
    axs2[6].set_yticklabels([])
    axs2[6].set_xticklabels([])
    
    ax5_divider = make_axes_locatable(axs2[6])
    cax5 = ax5_divider.append_axes("right", size="7%", pad="2%")
    cb5 = fig.colorbar(im_lfp1, cax=cax5) 
    ###
    im_lfp2 = axs2[7].imshow(np.repeat(np.tile(psd_gamma[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
    axs2[7].set_title('psd_gamma',fontsize = 30)
    # axs3[4].set_ylim([32 , 350])
    # axs3[4].set_xlim([0, 100])
    axs2[7].set_yticklabels([])
    axs2[7].set_xticklabels([])
    
    ax6_divider = make_axes_locatable(axs2[7])
    cax6 = ax6_divider.append_axes("right", size="7%", pad="2%")
    cb6 = fig.colorbar(im_lfp2, cax=cax6) 
    
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs2[8], label='right')
    
    ###
    
    bins = np.arange(20, 3840, 40)
    n, b, patches = plt.hist(z_reg[which], bins = bins);
    gaussian_smooth = gaussian_filter(n/180, sigma=1)
    axs1[7].plot(gaussian_smooth, (b[0:len(b) - 1] + b[1:None])/2)
    axs1[7].set_xlabel('firing rate (/s)',fontsize = 30)
    axs1[7].set_ylim([20, 3840])
    axs1[7].set_xlim([0, 40])
    
    axs1[8].set_axis_off()
    
    axs2[0].set_yticklabels([])
    axs2[1].set_yticklabels([])
    axs2[2].set_yticklabels([])
    axs2[3].set_yticklabels([])
    axs2[4].set_yticklabels([])
    axs2[5].set_yticklabels([])
    axs2[6].set_yticklabels([])
    axs2[7].set_yticklabels([])

    
    axs2[0].invert_yaxis()
    axs2[1].invert_yaxis()
    axs2[2].invert_yaxis()
    axs2[3].invert_yaxis()
    axs2[4].invert_yaxis()
    axs2[5].invert_yaxis()
    axs2[6].invert_yaxis()
    # axs2[7].invert_yaxis()
    
    # axs1[0].set_ylim([320 , 3500])
    # axs1[1].set_ylim([320 , 3500])
    # axs1[2].set_ylim([320 , 3500])
    # axs1[3].set_ylim([320 , 3500])
    # axs1[4].set_ylim([320 , 3500])
    # axs1[5].set_ylim([320 , 3500])
    # axs1[6].set_ylim([320 , 3500])
    
#     ###################
#     im_ap1 = axs2[0].imshow(peak_value_bin_means[:,None], cmap=mpl.colormaps['RdBu'], extent=[0,100,3500,320], aspect='auto', vmin = -10, vmax = -2)
#     axs2[0].set_title('peak_value')
#     axs2[0].invert_yaxis()
#     axs2[0].set_yticklabels([])
    
#     ax1_divider = make_axes_locatable(axs2[0])
#     cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
#     cb1 = fig.colorbar(im_ap1, cax=cax1)
    
#     ###
    
#     im_ap2 = axs2[1].imshow(halfpeak_duration_bin_means[:,None], cmap=mpl.colormaps['viridis'], extent=[0,100,3500,320], aspect='auto', vmin = 0.12, vmax = 0.16)
#     axs2[1].set_title('halfpeak_duration')
#     axs2[1].invert_yaxis()
#     axs2[1].set_yticklabels([])
    
#     ax2_divider = make_axes_locatable(axs2[1])
#     cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
#     cb2 = fig.colorbar(im_ap2, cax=cax2)
    
#     ###
    
#     im_ap3 = axs2[2].imshow(ptp_duration_bin_means[:,None], cmap=mpl.colormaps['viridis'], extent=[0,100,3500,320], aspect='auto', vmin = 0.25, vmax = 0.4)
#     axs2[2].set_title('ptp_duration')
#     axs2[2].invert_yaxis()
#     axs2[2].set_yticklabels([])
    
#     ax3_divider = make_axes_locatable(axs2[2])
#     cax3 = ax3_divider.append_axes("right", size="7%", pad="2%")
#     cb3 = fig.colorbar(im_ap3, cax=cax3)
    
#     ###
#     if i == 2:
#         im_ap4 = axs2[3].imshow(peak_trough_ratio_bin_means[:,None], cmap=mpl.colormaps['RdBu'], extent=[0,100,3500,320], aspect='auto', vmin = -1.2, vmax = 1.2)

#     else:
#         im_ap4 = axs2[3].imshow(peak_trough_ratio_bin_means[:,None], cmap=mpl.colormaps['RdBu'], extent=[0,100,3500,320], aspect='auto', vmin = -1.2, vmax = -0.8)
#     axs2[3].set_title('peak_trough_ratio')
#     axs2[3].invert_yaxis()
#     axs2[3].set_yticklabels([])
    
#     ax4_divider = make_axes_locatable(axs2[3])
#     cax4 = ax4_divider.append_axes("right", size="7%", pad="2%")
#     cb4 = fig.colorbar(im_ap4, cax=cax4)  
    
#     ###
    
#     im_ap5 = axs2[4].imshow(spatial_spread_bin_means[:,None], cmap=mpl.colormaps['viridis'], extent=[0,100,3500,320], aspect='auto', vmin = 10, vmax = 40)
#     axs2[4].set_title('spatial_spread')
#     axs2[4].invert_yaxis()
#     axs2[4].set_yticklabels([])
    
#     ax5_divider = make_axes_locatable(axs2[4])
#     cax5 = ax5_divider.append_axes("right", size="7%", pad="2%")
#     cb5 = fig.colorbar(im_ap5, cax=cax5)   
        
#     ###
    
#     im_ap6 = axs2[5].imshow(maxptps_bin_means[:,None], cmap=mpl.colormaps['viridis'], extent=[0,100,3500,320], aspect='auto', vmin = 8, vmax = 14)
#     axs2[5].set_title('maxptps')
#     axs2[5].invert_yaxis()
#     axs2[5].set_yticklabels([])
    
#     ax6_divider = make_axes_locatable(axs2[5])
#     cax6 = ax6_divider.append_axes("right", size="7%", pad="2%")
#     cb6 = fig.colorbar(im_ap6, cax=cax6) 
    
#     ###
#     plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
#                            brain_regions=br, display=True, ax=axs2[6], title='Real')
#     # axs2[0].set_ylim([32 , 350])
#     # axs2[1].set_ylim([32 , 350])
#     # axs2[2].set_ylim([32 , 350])
#     # axs2[3].set_ylim([32 , 350])
#     # axs2[4].set_ylim([32 , 350])
#     axs2[6].set_ylim([320 , 3500])
    
#     ###################
    
#     psd_delta = df_voltage['psd_delta'][pID].to_numpy()
#     psd_theta = df_voltage['psd_theta'][pID].to_numpy()
#     psd_alpha = df_voltage['psd_alpha'][pID].to_numpy()
#     psd_beta = df_voltage['psd_beta'][pID].to_numpy()
#     psd_gamma = df_voltage['psd_gamma'][pID].to_numpy()
    
    
#     im_lfp1 = axs3[0].imshow(np.repeat(np.tile(psd_delta[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
#     axs3[0].set_title('psd_delta')
#     # axs3[0].set_ylim([32 , 350])
#     # axs3[0].set_xlim([0, 100])
#     # axs3[0].set_yticklabels([])
#     axs3[0].set_xticklabels([])
    
#     im_lfp2 = axs3[1].imshow(np.repeat(np.tile(psd_theta[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
#     axs3[1].set_title('psd_theta')
#     # axs3[1].set_ylim([32 , 350])
#     # axs3[1].set_xlim([0, 100])
#     axs3[1].set_yticklabels([])
#     axs3[1].set_xticklabels([])
    
#     im_lfp3 = axs3[2].imshow(np.repeat(np.tile(psd_alpha[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
#     axs3[2].set_title('psd_alpha')
#     # axs3[2].set_ylim([32 , 350])
#     # axs3[2].set_xlim([0, 100])
#     axs3[2].set_yticklabels([])
#     axs3[2].set_xticklabels([])    
    
#     im_lfp4 = axs3[3].imshow(np.repeat(np.tile(psd_beta[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
#     axs3[3].set_title('psd_beta')
#     # axs3[3].set_ylim([32 , 350])
#     # axs3[3].set_xlim([0, 100])
#     axs3[3].set_yticklabels([])
#     axs3[3].set_xticklabels([]) 
    
#     im_lfp5 = axs3[4].imshow(np.repeat(np.tile(psd_gamma[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
#     axs3[4].set_title('psd_gamma')
#     # axs3[4].set_ylim([32 , 350])
#     # axs3[4].set_xlim([0, 100])
#     axs3[4].set_yticklabels([])
#     axs3[4].set_xticklabels([])
    
#     plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
#                            brain_regions=br, display=True, ax=axs3[5], title='Real', label='right')

    
    
#     axs3[6].set_axis_off()

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


    # which = max_ptps>6 

    # ptp_duration = np.load(out_dir + '/ptp_duration.npy')
    # peak_trough_ratio = np.load(out_dir + '/peak_trough_ratio.npy')
    # spatial_spread = np.load(out_dir + '/spatial_spread_th12.npy')
    # reploarization_slope = np.load(out_dir + '/reploarization_slope_window_50.npy')
    # recovery_slope = np.load(out_dir + '/recovery_slope.npy')
    # spatial_non_threshold = np.load(out_dir + '/non_threshold_spatial_spread.npy')

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


    data = regional_ptp_duration
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[2], showextrema=False)
    axs[2].set_ylabel('ptp_duration',fontsize = 30)


    data = regional_peak_trough_ratio
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[0], showextrema=False)
    axs[0].set_ylabel('peak_trough_ratio',fontsize = 30)

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
    sns.violinplot(data=df, ax = axs[4], showextrema=False)
    axs[4].set_ylabel('reploarization_slope',fontsize = 30)

    data = regional_recovery_slope
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[5], showextrema=False)
    axs[5].set_ylabel('recovery_slope',fontsize = 30)

    data = regional_spatial_non_threshold
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[3], showextrema=False)
    axs[3].set_ylabel('spatial_spread',fontsize = 30)

    data = regional_maxptps
    maxsize = max([a.size for a in data.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-v.size,), mode='constant', constant_values=np.nan) for k,v in data.items()}
    df = pd.DataFrame(data_pad)
    sns.violinplot(data=df, ax = axs[1], showextrema=False)
    axs[1].set_ylabel('max_ptps',fontsize = 30)
    
    fig.suptitle(pID, fontsize=40)

    plt.savefig(main_dir + '/region_feature_' + eID + '_' + probe + '_ptp_th_6.png')
    # plt.savefig(out_dir + '/region_feature_' + eID + '_' + probe + '_ptp_th_8.png')

# %%
bins = np.arange(20, 3840, 40)
n, b, patches = plt.hist(z_reg, bins = bins);
gaussian_smooth = gaussian_filter(n, sigma=1)
plt.plot((b[0:len(b) - 1] + b[1:None])/2,gaussian_smooth)

# %%
plt.hist(ptp_duration[which]/300, bins = np.arange(0, 0.8, 0.05))

# %% jupyter={"outputs_hidden": true}
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets'
bin_size = 40
z_reg_bins = np.arange(320, 3500, bin_size)
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    out_dir = main_dir + '/' + eID + '_' + probe
    h5_path = out_dir + '/' + 'subtraction.h5'    
    try:
        h5 = h5py.File(h5_path)

        x = h5["localizations"][:,0]
        z_reg = h5["localizations"][:,2]
        maxptps = h5["maxptps"][:]
    except:
        continue
    
    h5.close()
    print(i)
    try:
        peak_value = np.load(out_dir + '/peak_value.npy')
        ptp_duration = np.load(out_dir + '/ptp_duration.npy')
        halfpeak_duration = np.load(out_dir + '/halfpeak_duration.npy')
        peak_trough_ratio = np.load(out_dir + '/peak_trough_ratio.npy')
        spatial_spread = np.load(out_dir + '/spatial_spread.npy')
        velocity = np.load(out_dir + '/velocity_th_25.npy')
        ci_err = np.load(out_dir + '/velocity_ci_th_25.npy')
    except:
        continue
    


    which = maxptps>8

    # fig, [axs1, axs2, axs3] = plt.subplots(3, 7, figsize = (24, 60))
    fig, [axs1, axs2] = plt.subplots(2, 7, figsize = (24, 40))

    cmps = np.clip(peak_value[which], -13, 13)
    
    
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
    
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
    

    br = BrainRegions()


    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)


    
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs1[4], title='Real', label='right')
    axs1[0].set_ylim([320 , 3500])
    axs1[1].set_ylim([320 , 3500])
    axs1[2].set_ylim([320 , 3500])
    axs1[3].set_ylim([320 , 3500])
    axs1[4].set_ylim([320 , 3500])
    
    
    axs1[5].set_axis_off()
    axs1[6].set_axis_off()
    # axs1[5].set_ylim([320 , 3500])
    # axs1[6].set_ylim([320 , 3500])
    
    ###################

    
    
    im_ap1 = axs2[0].imshow(ptp_duration_bin_means[:,None], cmap=mpl.colormaps['viridis'], extent=[0,100,3500,320], aspect='auto', vmin = 0.25, vmax = 0.4)
    axs2[0].set_title('ptp_duration')
    axs2[0].invert_yaxis()
    axs2[0].set_yticklabels([])
    
    ax1_divider = make_axes_locatable(axs2[0])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(im_ap1, cax=cax1)
    
    ###
    if i == 2:
        im_ap2 = axs2[1].imshow(peak_trough_ratio_bin_means[:,None], cmap=mpl.colormaps['RdBu'], extent=[0,100,3500,320], aspect='auto', vmin = -1.2, vmax = 1.2)

    else:
        im_ap2 = axs2[1].imshow(peak_trough_ratio_bin_means[:,None], cmap=mpl.colormaps['RdBu'], extent=[0,100,3500,320], aspect='auto', vmin = -1.2, vmax = -0.8)
    axs2[1].set_title('peak_trough_ratio')
    axs2[1].invert_yaxis()
    axs2[1].set_yticklabels([])
    
    ax2_divider = make_axes_locatable(axs2[1])
    cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
    cb2 = fig.colorbar(im_ap2, cax=cax2)  
    
    ###
    
    im_ap3 = axs2[2].imshow(spatial_spread_bin_means[:,None], cmap=mpl.colormaps['viridis'], extent=[0,100,3500,320], aspect='auto', vmin = 10, vmax = 40)
    axs2[2].set_title('spatial_spread')
    axs2[2].invert_yaxis()
    axs2[2].set_yticklabels([])
    
    ax3_divider = make_axes_locatable(axs2[2])
    cax3 = ax3_divider.append_axes("right", size="7%", pad="2%")
    cb3 = fig.colorbar(im_ap3, cax=cax3)   
        
    ###
    
    im_ap4 = axs2[3].imshow(maxptps_bin_means[:,None], cmap=mpl.colormaps['viridis'], extent=[0,100,3500,320], aspect='auto', vmin = 10, vmax = 16)
    axs2[3].set_title('maxptps')
    axs2[3].invert_yaxis()
    axs2[3].set_yticklabels([])
    
    ax4_divider = make_axes_locatable(axs2[3])
    cax4 = ax4_divider.append_axes("right", size="7%", pad="2%")
    cb4 = fig.colorbar(im_ap4, cax=cax4) 
    ###
    psd_theta = df_voltage['psd_theta'][pID].to_numpy()
    psd_gamma = df_voltage['psd_gamma'][pID].to_numpy()
    
    im_lfp1 = axs2[4].imshow(np.repeat(np.tile(psd_theta[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
    axs2[4].set_title('psd_theta')
    # axs3[1].set_ylim([32 , 350])
    # axs3[1].set_xlim([0, 100])
    axs2[4].set_yticklabels([])
    axs2[4].set_xticklabels([])
    
    ax5_divider = make_axes_locatable(axs2[4])
    cax5 = ax5_divider.append_axes("right", size="7%", pad="2%")
    cb5 = fig.colorbar(im_lfp1, cax=cax5) 
    ###
    im_lfp2 = axs2[5].imshow(np.repeat(np.tile(psd_gamma[:,None],[1, 100]),10, axis = 0), aspect='auto', origin = 'lower')
    axs2[5].set_title('psd_gamma')
    # axs3[4].set_ylim([32 , 350])
    # axs3[4].set_xlim([0, 100])
    axs2[5].set_yticklabels([])
    axs2[5].set_xticklabels([])
    
    ax6_divider = make_axes_locatable(axs2[5])
    cax6 = ax6_divider.append_axes("right", size="7%", pad="2%")
    cb6 = fig.colorbar(im_lfp2, cax=cax6) 
    ###
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs2[6], title='Real', label='right')
    axs2[0].set_ylim([320 , 3500])
    axs2[1].set_ylim([320 , 3500])
    axs2[2].set_ylim([320 , 3500])
    axs2[3].set_ylim([320 , 3500])
    axs2[4].set_ylim([320 , 3500])
    axs2[4].set_ylim([320 , 3500])
    axs2[6].set_ylim([320 , 3500])
    
    ###################
    
    
    plt.savefig(out_dir + '/region_feature_' + eID + '_' + probe + '_ptp_th_8.png')

# %%
np.shape(np.repeat(np.tile(psd_beta[:,None],[1, 100]),10, axis = 0))

# %%
plt.imshow(psd_delta[:, None], aspect='auto')

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
np.max(x)

# %%
np.min(z_reg)

# %%
np.max(z_reg)

# %%
h5 = h5py.File(h5_path)
geom = h5['geom'][:]
h5.close()

# %%
np.min(geom[:,0])
np.max(geom[:,1])

# %%
np.max(geom[:,0])

# %%
import h5py
import numpy as np
h5_sub = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets/4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a_probe01/subtraction.h5'

with h5py.File(h5_sub) as h5:
    waveform = h5["denoised_waveforms"][:]

# %%
# import matplotlib.pyplot as plt
i = 1
win = 5
max_idx  = np.nanargmax(np.ptp(waveform[i],axis = 0))

peak_idx  = np.nanargmax(waveform[i][:,max_idx])
trough_idx  = np.nanargmin(waveform[i][:,max_idx])

peak = np.nanmax(waveform[i][:,max_idx])
trough  = np.nanmin(waveform[i][:,max_idx])

if np.abs(peak)>np.abs(trough):
    plt.plot(-waveform[i][:,max_idx])
    # plt.vlines(trough_idx, -peak, -trough, 'r')
    # plt.vlines(trough_idx + win, -peak, -trough,'r')
    plt.vlines(peak_idx, -peak, -trough, 'r')
    plt.vlines(peak_idx + win, -peak, -trough,'r')
else:
    plt.plot(waveform[i][:,max_idx])
    # plt.vlines(peak_idx, trough, peak,'r')
    # plt.vlines(peak_idx + win, trough, peak,'r')
    plt.vlines(trough_idx, trough, peak,'r')
    plt.vlines(trough_idx + win, trough, peak,'r')

# %%
import h5py
sub_h51 = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_fece187f-b47f-4870-a1d6-619afe942a7d_probe_probe01_pID_5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e/subtraction.h5'
with h5py.File(sub_h51) as h5:
    z_abs1 = h5["localizations"][:, 2]
    x1 = h5["localizations"][:, 0]
    y1 = h5["localizations"][:, 1]
    times1 = h5["spike_index"][:, 0] / 30_000
    maxptps1 = h5["maxptps"][:]

sub_h52 = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets/fece187f-b47f-4870-a1d6-619afe942a7d_probe01/subtraction.h5'
with h5py.File(sub_h52) as h5:
    z_abs2 = h5["localizations"][:, 2]
    x2 = h5["localizations"][:, 0]
    y2 = h5["localizations"][:, 1]
    times2 = h5["spike_index"][:, 0] / 30_000
    maxptps2 = h5["maxptps"][:]


# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize = [20, 20])
axs[0].scatter(times1, z_abs1, c = maxptps1, s = 0.1)
axs[0].set_xlim([0, 180])
axs[1].scatter(times2, z_abs2, c = maxptps2, s = 0.1)
axs[1].set_xlim([0, 180])

# %%
plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, label='right')

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
from one.api import ONE
one = ONE(base_url="https://alyx.internationalbrainlab.org")

old_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_datasets'
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    
    old_out_dir = old_dir + '/' + eID + '_' + probe
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    
    sub_h51 = out_dir + '/subtraction.h5'
    with h5py.File(sub_h51) as h5:
        z_abs1 = h5["localizations"][:, 2]
        x1 = h5["localizations"][:, 0]
        y1 = h5["localizations"][:, 1]
        times1 = h5["spike_index"][:, 0] / 30_000
        maxptps1 = h5["maxptps"][:]
        
    which = maxptps1>6
    plt.figure(constrained_layout=True)
    fig, axs = plt.subplots(1, 3, figsize = [30, 8], width_ratios = [8,1,1])
    
    axs[0].scatter(times1[which], z_abs1[which], c = maxptps1[which], s = 0.1)
    axs[0].set_xlim([0, 180])
    axs[0].set_ylim([20, 3840])
    axs[0].set_title('raster across time', fontsize = 40)
    
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))

    br = BrainRegions()

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)
    
    
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[1], label='right')
    
    
    bins = np.arange(20, 3840, 40)
    n, b = np.histogram(z_abs1[which], bins = bins);
    gaussian_smooth = gaussian_filter(n/180, sigma=1)
    axs[2].plot(gaussian_smooth, (b[0:len(b) - 1] + b[1:None])/2)
    axs[2].set_xlabel('firing rate (/s)',fontsize = 30)
    axs[2].set_ylim([20, 3840])
    # axs[2].set_xlim([0, 40])
    # try:
    #     sub_h52 = old_out_dir + '/subtraction.h5'
    #     with h5py.File(sub_h52) as h5:
    #         z_abs2 = h5["localizations"][:, 2]
    #         x2 = h5["localizations"][:, 0]
    #         y2 = h5["localizations"][:, 1]
    #         times2 = h5["spike_index"][:, 0] / 30_000
    #         maxptps2 = h5["maxptps"][:]
    #     axs[1].scatter(times2, z_abs2, c = maxptps2, s = 0.1)
    #     axs[1].set_xlim([0, 180])
    #     axs[1].set_title('spike interface destripe')
    # except:
    #     continue
        
    
    
    plt.savefig(main_dir + '/raster_destripe_' + pID + '.png')

# %%
rec = spikeinterface.core.binaryrecordingextractor.BinaryRecordingExtractor('/local/monkey_insertion/destriped_pacman-task_c_220218_neu_insertion_g0_t0.imec0.ap.bin', 30_000, 384, 'float32')


# %%
