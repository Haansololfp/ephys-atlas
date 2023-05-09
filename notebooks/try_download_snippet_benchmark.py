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
from pathlib import Path
from one.api import ONE
from one.remote import aws

# %% jupyter={"outputs_hidden": true}
# http://benchmarks.internationalbrainlab.org.s3-website-us-east-1.amazonaws.com/#/0/4

LOCAL_DATA_PATH = Path.home().joinpath('/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example')

one = ONE(base_url='https://alyx.internationalbrainlab.org')
s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)


pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
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
    'fe380793-8035-414e-b000-09bfe5ece92a',
]

# downloads all pids (52 Gb total)
if False:
    aws.s3_download_folder("resources/ephys-atlas-sample", LOCAL_DATA_PATH, s3=s3, bucket_name=bucket_name)

# downloads one pid at a time (3 to 7 Gb a pop)
if True:
    for i in range(len(pids)):
        pid = pids[i]
        aws.s3_download_folder(f"resources/ephys-atlas-sample/{pid}", LOCAL_DATA_PATH.joinpath(pid), s3=s3, bucket_name=bucket_name)

# %%
import numpy as np
import matplotlib.pyplot as plt
from brainbox.ephys_plots import plot_brain_regions
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ibllib.atlas import BrainRegions

# %%
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

# %% jupyter={"outputs_hidden": true}
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example'

br = BrainRegions()
for i in range(len(pids)):
    pid = pids[i]

    ap_path = main_dir + '/' + pid + "/T00500/ap.npy"
    ap_0500 = np.load(ap_path)

    ap_path = main_dir + '/' + pid + "/T02500/ap.npy"
    ap_2500 = np.load(ap_path)
    long_enough = True
    try:
        ap_path = main_dir + '/' + pid + "/T04500/ap.npy"
        ap_4500 = np.load(ap_path)
    except:
        ap_path = main_dir + '/' + pid + "/T01500/ap.npy"
        ap_1500 = np.load(ap_path)
        long_enough = False
        
    
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pid]
    df_channels = df_channels.reset_index(drop=True)
    channel_ids = df_channels['atlas_id'].values


    plt.figure(figsize = (20,20))

    fig, axs = plt.subplots(3, 1, figsize = (20,40))

    ts = 1000
    chunk = 1000
    axs[0].imshow(ap_0500[:,ts:(ts+chunk)], cmap = 'RdBu', vmin = -0.0001, vmax = 0.0001, aspect='auto')
    axs[0].invert_yaxis()
    axs[0].set_title('T0500')

    axs1_divider = make_axes_locatable(axs[0])
    reg_axs1 = axs1_divider.append_axes("right", size="10%", pad="2%")
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=reg_axs1, label='right')
    

    axs[1].imshow(ap_2500[:,ts:(ts+chunk)], cmap = 'RdBu', vmin = -0.0001, vmax = 0.0001, aspect='auto')
    axs[1].invert_yaxis()
    axs[1].set_title('T2500')

    axs2_divider = make_axes_locatable(axs[1])
    reg_axs2 = axs2_divider.append_axes("right", size="10%", pad="2%")
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=reg_axs2, label='right')
    
    
    if long_enough:
        axs[2].imshow(ap_4500[:,ts:(ts+chunk)], cmap = 'RdBu', vmin = -0.0001, vmax = 0.0001, aspect='auto')
        axs[2].invert_yaxis()
        axs[2].set_title('T4500')
    else:
        axs[2].imshow(ap_1500[:,ts:(ts+chunk)], cmap = 'RdBu', vmin = -0.0001, vmax = 0.0001, aspect='auto')
        axs[2].invert_yaxis()
        axs[2].set_title('T1500')

    axs3_divider = make_axes_locatable(axs[2])
    reg_axs3 = axs3_divider.append_axes("right", size="10%", pad="2%")
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=reg_axs3, label='right')
    
    plt.suptitle(pid, fontsize=24)
    
    plt.savefig(main_dir + '/' + pid + '_waveform_shape.png')

# %%
wfs_path = main_dir + '/' + pid + "/T00500/waveforms.npy"
wfs_0500 = np.load(wfs_path)

# %%
np.shape(wfs_0500)

# %%
plt.plot(wfs_0500[10000]);

# %%
from spike_psvae import denoise
denoiser_init_kwargs = {}
denoiser = denoise.SingleChanDenoiser(**denoiser_init_kwargs)

# %%
# from spike_psvae.denoise import SingleChanDenoiser
# import torch
dn = SingleChanDenoiser().load()
wfs = np.swapaxes(wfs_0500, 1, 2)
wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)

# %%
wfs_denoised = wfs_denoised.detach().numpy()

# %%
bias = np.arange(40)

# %%
np.shape(bias[:,None])

# %%
# bias = np.repeat(bias[:,None], 121, axis =1 )
plt.plot(wfs_0500[10000]+bias.T, c = 'k');
plt.plot((wfs_denoised[10000]+bias).T, '--', c = 'r');

# %%
df_spikes = pd.read_parquet('/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example/' + pid + '/T03500/spikes.pqt')

# %%
df_spikes['z']

# %%
ap_spikes = np.load('/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example/' + pid + '/T03500/waveforms.npy')

# %%
np.shape(ap_spikes)

# %%
from ephys_atlas.data import download_tables
LABEL = '2022_W34'
LABEL = '2023_W14'
LOCAL_DATA_PATH = Path("/mnt/s1/ephys-atlas-decoding/tables")
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')

df_raw_features, df_clusters, df_channels = download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)

# %%
