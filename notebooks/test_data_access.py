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
import spikeinterface.preprocessing as si
import spikeinterface.extractors as se

# %%
from one.api import ONE

# %%
one = ONE(base_url="https://alyx.internationalbrainlab.org")

# %%
sessions = one.alyx.rest('sessions', 'list', tag='2022_Q2_IBL_et_al_RepeatedSite')

# %%
eids = one.to_eid(sessions)

# %%
session, probe = one.pid2eid('1a276285-8b0e-4cc9-9f0a-a3a002978724')

# %%
dataset_contents = one.list_datasets(eid='69c9a415-f7fa-4208-887b-1417c1479b48', collection="raw_ephys_data/*")
raw_contents = [dataset_content for dataset_content in dataset_contents if not dataset_content.endswith(".npy")]
probe_labels = set([raw_content.split("/")[1] for raw_content in raw_contents])

# %%
rec.get_stream_names

# %% jupyter={"outputs_hidden": true}
one.list_datasets(eid='69c9a415-f7fa-4208-887b-1417c1479b48', collection="raw_ephys_data/*")

# %%
rec = se.read_ibl_streaming_recording(
    '69c9a415-f7fa-4208-887b-1417c1479b48',
    first_ap_stream,
    cache_folder="/local/sicache",
)

# %%
one.eid2pid('69c9a415-f7fa-4208-887b-1417c1479b48')

# %%
rec = se.read_ibl_streaming_recording(
    session['id'],
    first_ap_stream,
    cache_folder="/local/sicache",
)

# %% jupyter={"outputs_hidden": true}
one.alyx.rest('sessions', 'list', eid )

# %%
from spike_psvae import subtract

# %% jupyter={"outputs_hidden": true}
subcache = '/moto/stats/users/hy2562/projects/ephys_atlas/test'

first_ap_stream = next(sn for sn in se.IblStreamingRecordingExtractor.get_stream_names(session=session['id']) if sn.endswith(".ap"))

rec = se.read_ibl_streaming_recording(
    session['id'],
    first_ap_stream,
    cache_folder="/local/sicache",
)
fs = int(rec.get_sampling_frequency())

rec = rec.frame_slice(start_frame=int(2000*fs),end_frame=int(2180*fs))

rec = si.highpass_filter(rec)
rec = si.phase_shift(rec)
bad_channel_ids, channel_labels = si.detect_bad_channels(rec, num_random_chunks=100)
print(f"{bad_channel_ids=}")
rec = si.interpolate_bad_channels(rec, bad_channel_ids)
rec = si.highpass_spatial_filter(rec)
# we had been working with this before -- should switch to MAD,
# but we need to rethink the thresholds
rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100)
print(rec)
# /local is too small
# rec = rec.save_to_folder(folder=ppxcache)

# if subcache.exists():
#     shutil.rmtree(subcache)
sub_h5 = subtract.subtraction(
    rec,
    out_folder=subcache,
    thresholds=[12, 10, 8, 6],#[12, 10, 8, 6, 5],
    n_sec_pca=40,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    save_denoised_waveforms=True,
    n_jobs=14,
    loc_workers=1,
    overwrite=False,
)

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/test'
out_dir = main_dir + '/' + session['id'] + '_' + first_ap_stream[:-3]

# %%
session_names

# %%
rec = se.read_ibl_streaming_recording(
        eID,
        'probe01.ap',
        cache_folder="/local/sicache",
    )

rec

# %%
first_ap_stream

# %%
first_ap_stream

# %%
rec = se.read_ibl_streaming_recording(
    eID,
    first_ap_stream,
    load_sync_channel=True,
    cache_folder="/local/sicache",
)
rec

# %% jupyter={"outputs_hidden": true}
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/test'
for i in range(10):
    session = sessions[i]
    out_dir = main_dir + '/' + session['id'] + '_' + first_ap_stream[:-3]
    # !mkdir {out_dir}
    first_ap_stream = next(sn for sn in se.IblStreamingRecordingExtractor.get_stream_names(session=session['id']) if sn.endswith(".ap"))

    rec = se.read_ibl_streaming_recording(
        session['id'],
        first_ap_stream,
        cache_folder="/local/sicache",
    )
    fs = int(rec.get_sampling_frequency())

    rec = rec.frame_slice(start_frame=int(2000*fs),end_frame=int(2180*fs))

    rec = si.highpass_filter(rec)
    rec = si.phase_shift(rec)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec, num_random_chunks=100)
    print(f"{bad_channel_ids=}")
    rec = si.interpolate_bad_channels(rec, bad_channel_ids)
    rec = si.highpass_spatial_filter(rec)
    # we had been working with this before -- should switch to MAD,
    # but we need to rethink the thresholds
    rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100)
    print(rec)
    # /local is too small
    # rec = rec.save_to_folder(folder=ppxcache)

    # if subcache.exists():
    #     shutil.rmtree(subcache)
    sub_h5 = subtract.subtraction(
        rec,
        out_folder=out_dir,
        thresholds=[12, 10, 8, 6],#[12, 10, 8, 6, 5],
        n_sec_pca=40,
        save_subtracted_tpca_projs=False,
        save_cleaned_tpca_projs=False,
        save_denoised_tpca_projs=True,
        # save_denoised_waveforms=True,
        n_jobs=14,
        loc_workers=1,
        overwrite=False,
    )


# %%
from spike_psvae import ibme

# %%
import h5py

# %%
sub_h5 = '/moto/stats/users/hy2562/projects/ephys_atlas/test/e2b845a1-e313-4a08-bc61-a5f662ed295e_probe00/subtraction.h5'
with h5py.File(sub_h5) as h5:
    z_abs = h5["localizations"][:, 2]
    x = h5["localizations"][:, 0]
    y = h5["localizations"][:, 1]
    times = (h5["spike_index"][:, 0] + 30_000*2000) / 30_000
    maxptps = h5["maxptps"][:]

# %%
eID = 'e2b845a1-e313-4a08-bc61-a5f662ed295e'

pID = one.eid2pid(eID)
idx = pID[1].index('probe00')
pID = pID[0][idx]
channel_region = df_channels['acronym'][pID]

# %%
df_channels['axial_um'][pID]

# %%
reg = channel_region[0:384:2]

# %%
a,b = np.unique(reg, return_index=True)

# %%
import matplotlib.cm as cm
a = cm.get_cmap('tab20b', 5)
colors = [a.colors[i] for i in range(len(a.colors))]

# %%
labels

# %%

plt.figure(figsize=(4, 20))
reg = channel_region[0:384:2]
heights = df_channels['axial_um'][pID]
heights = heights[0:384:2]

labels = np.unique(reg)
a = cm.get_cmap('tab20b', len(labels))
colors = np.array([a.colors[i] for i in range(len(a.colors))])

# reg = reg.replace(labels, np.arange(len(labels)))

# colors = colors[np.int16(reg.to_numpy()),0:3]
# plt.imshow(np.int16(reg.to_numpy())[:,None], cmap = 'tab20b')


# plt.bar(np.arange(len(heights)), heights, color=colors)

# %% jupyter={"outputs_hidden": true}
# import matplotlib.pyplot as plt
br = BrainRegions()
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/clusters.pqt'))
df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/probes.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/depths.pqt'))
df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/raw_ephys_features.pqt'))

for i in range(0,len(subfolders)):
    
    s_eID = subfolders[i].rfind('/') + 1
    e_eID = subfolders[i].rfind('_')
    eID = subfolders[i][s_eID:e_eID]
    if eID[0]=='.':
        continue
        
    sub_h5 = subfolders[i] + '/subtraction.h5'
    with h5py.File(sub_h5) as h5:
        z_abs = h5["localizations"][:, 2]
        x = h5["localizations"][:, 0]
        y = h5["localizations"][:, 1]
        times = (h5["spike_index"][:, 0] + 30_000*2000) / 30_000
        maxptps = h5["maxptps"][:]


    pID = one.eid2pid(eID)
    idx = pID[1].index('probe00')
    pID = pID[0][idx]
    
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
    
    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)


    fig, ax = plt.subplots(1, 3, figsize=(22, 20), width_ratios = [1, 10, 1])
    cmps = np.clip(maxptps, 3, 13)
    nmps = 0.25 + 0.74 * (cmps - cmps.min()) / (cmps.max() - cmps.min())
    ax[0].scatter(x, z_abs, c=cmps, alpha=nmps, s=0.1)
    ax[0].axis("on")

    ax[1].scatter(times, z_abs, c=cmps, alpha=nmps, s=0.1)
    ax[1].axis("on")
    ax[1].set_xlim([2000, 2180])
    
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=ax[2], title='Real')



#     ax[2].scatter(z_abs, c=cmps, alpha=nmps, s=0.1)
#     ax[2].axis("on")
    
    plt.savefig(eID + 'density_clouds.png')

# %%
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))

# %%
# pID = '1a276285-8b0e-4cc9-9f0a-a3a002978724'


pID = one.eid2pid(eID)
idx = pID[1].index('probe00')
pID = pID[0][idx]

df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))

df_channels = df_channels.reset_index(drop=False)
df_channels = df_channels[df_channels.pid == pID]
df_channels = df_channels.reset_index(drop=True)

# %%
df_channels['axial_um'].values

# %%
from ibllib.atlas import BrainRegions
from brainbox.ephys_plots import plot_brain_regions
import scipy
import pandas as pd

LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/clusters.pqt'))
df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/probes.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/depths.pqt'))
df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/raw_ephys_features.pqt'))




pID = one.eid2pid(eID)
idx = pID[1].index('probe00')
pID = pID[0][idx]

br = BrainRegions()


df_channels = df_channels.reset_index(drop=False)
df_channels = df_channels[df_channels.pid == pID]
df_channels = df_channels.reset_index(drop=True)


plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=None, title='Real')

# cberyl = scipy.interpolate.interp1d(df_pid['axial_um'].values, predictions_remap_beryl, kind='nearest', fill_value="extrapolate")(depths)
# plot_brain_regions(cberyl, channel_depths=depths, brain_regions=regions, display=True, ax=None, linewidth=0)

# %%
df_channels['atlas_id'][pID]

# %%
import os
directory = '/moto/stats/users/hy2562/projects/ephys_atlas/test'
sub_dir = os.walk(directory)

# %%
subfolders = [ f.path for f in os.scandir(directory) if f.is_dir() ]

# %%
import pandas as pd
from pathlib import Path
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('latest/channels.pqt'))
for i in range(len(subfolders)):
    s_eID = subfolders[i].rfind('/') + 1
    e_eID = subfolders[i].rfind('_')
    eID = subfolders[i][s_eID:e_eID]
    if eID[0]=='.':
        continue
    pID = one.eid2pid(eID)
    idx = pID[1].index('probe00')
    pID = pID[0][idx]
    try:
        channel_region = df_channels['acronym'][pID]
        print(eID)
        print(channel_region[0])
    except:
        channel_region = one.load_dataset(eID, 'channels.brainLocationIds_ccf_2017', collection = "alf/probe00/pykilosort")
        print(eID)
        print(channel_region[0])

# %%
h5_path = '/moto/stats/users/hy2562/projects/ephys_atlas/test/e2b845a1-e313-4a08-bc61-a5f662ed295e_probe00/subtraction.h5'
h5 = h5py.File(h5_path)
for k in h5:
    print(" - ", k) #, h5[k].shape

# %%
