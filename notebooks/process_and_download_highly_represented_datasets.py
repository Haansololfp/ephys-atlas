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

# %% [markdown]
# This is process and save a snippet of a set of similar datasets

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np

Benchmark_pids = np.load('/moto/stats/users/hy2562/projects/ephys_atlas/code/small_set_pids.npy', allow_pickle=True)

# %%
Benchmark_pids

# %%
from brainbox.io.spikeglx import Streamer
import sys
import subprocess
import spikeinterface.full as sf
from one.api import ONE
from pathlib import Path
import fileinput
one = ONE(base_url="https://alyx.internationalbrainlab.org")

webclient = one.alyx
cache_folder = "/moto/stats/users/hy2562/projects/ephys_atlas/cache"
destripe_py_dir = '/moto/stats/users/hy2562/spike-psvae/scripts/destripe.py'
save_folder = "/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/"

stream_type = "ap"
remove_cached = True

# %%
from brainbox.io.spikeglx import Streamer
import sys
import subprocess
import spikeinterface.full as sf
from one.api import ONE
from pathlib import Path
import fileinput
one = ONE(base_url="https://alyx.internationalbrainlab.org")

webclient = one.alyx
cache_folder = "/moto/stats/users/hy2562/projects/ephys_atlas/cache"
destripe_py_dir = '/moto/stats/users/hy2562/spike-psvae/scripts/destripe.py'
save_folder = "/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/"

stream_type = "ap"
remove_cached = True
for i in range(3,len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)

    ibl_treamer = Streamer(pid=pID, one=one, typ=stream_type, cache_folder=cache_folder, remove_cached=remove_cached)
    try:
        ibl_treamer._download_raw_partial(first_chunk=2000, last_chunk=2179)
    except:
        continue
    cbin_url = ibl_treamer.url_cbin
    
    alyx_base_path = one.eid2path(eID)
    relpath= alyx_base_path.relative_to(one.cache_dir)
    
    cbin_parent_dir = cache_folder + '/' + str(relpath) + '/raw_ephys_data/' + probe + '/chunk_002000_to_002179'
    
    cbin_dir = list(Path(cbin_parent_dir).glob('*.cbin'))[0]
    meta_dir = list(Path(cbin_parent_dir).glob('*.meta'))[0]
    ch_dir = list(Path(cbin_parent_dir).glob('*.ch'))[0]
    
    for line in fileinput.input(meta_dir, inplace=1):
        if 'fileTimeSecs' in line:
            line = 'fileTimeSecs=180\n'
        sys.stdout.write(line)
    
    subprocess.run(["python",
                    destripe_py_dir,
                    str(cbin_dir),
                   ],
                   check = True
                  )
    
    destriped_cbin_dir = list(Path(cbin_parent_dir).glob('destriped_*.cbin'))[0]
    destriped_meta_dir = list(Path(cbin_parent_dir).glob('destriped_*.meta'))[0]
    
    # destriped_cbin_dir = cbin_parent_dir + destriped_cbin_dir.name
    # destriped_meta_dir = cbin_parent_dir + destriped_meta_dir.name
    
    
    out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID + '/'
    # !mkdir {out_dir}
    
    # !mv {str(destriped_cbin_dir)} {out_dir}
    # !mv {str(destriped_meta_dir)} {out_dir}
    
    
    save_destriped_ch_dir = out_dir + 'destriped_' + ch_dir.name
     
    # !cp {ch_dir} {save_destriped_ch_dir}
    
    out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name
    
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    sub_h5 = subtract.subtraction(
                    rec,
                    out_folder=out_dir,
                    thresholds=[12, 10, 8, 6, 5],
                    n_sec_pca=40,
                    save_subtracted_tpca_projs=False,
                    save_cleaned_tpca_projs=False,
                    save_denoised_tpca_projs=False,
                    save_denoised_waveforms=True,
                    n_jobs=14,
                    loc_workers=1,
                    overwrite=False,
                    # n_sec_chunk=args.batchlen,
                    save_cleaned_pca_projs_on_n_channels=5,
                    loc_feature=("ptp", "peak"),
                )
        continue
    

# %% jupyter={"outputs_hidden": true}
from spike_psvae import subtract

main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'
for i in range(9, len(Benchmark_pids)):
    try:
        pID = Benchmark_pids[i]
        eID, probe = one.pid2eid(pID)

        out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

        destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
        destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

        rec_cbin = sf.read_cbin_ibl(Path(out_dir))
        rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
        rec.set_probe(rec_cbin.get_probe(), in_place=True)
        fs = rec.get_sampling_frequency()

        sub_h5 = subtract.subtraction(
            rec,
            out_folder=out_dir,
            thresholds=[12, 10, 8, 6, 5],
            n_sec_pca=40,
            save_subtracted_tpca_projs=False,
            save_cleaned_tpca_projs=False,
            save_denoised_tpca_projs=False,
            save_denoised_waveforms=True,
            n_jobs=14,
            loc_workers=1,
            overwrite=False,
            # n_sec_chunk=args.batchlen,
            #save_cleaned_pca_projs_on_n_channels=5,
            loc_feature=("ptp", "peak"),
        )
    except:
        continue


# %%
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

# %% jupyter={"outputs_hidden": true}
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from ibllib.atlas import BrainRegions
from brainbox.ephys_plots import plot_brain_regions
from scipy.ndimage import gaussian_filter
one = ONE(base_url="https://alyx.internationalbrainlab.org")

main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'
for i in range(1,len(Benchmark_pids)):
    try:
        pID = Benchmark_pids[i]
        eID, probe = one.pid2eid(pID)


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

        df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

        br = BrainRegions()

        df_channels = df_channels.reset_index(drop=False)
        df_channels = df_channels[df_channels.pid == pID]
        df_channels = df_channels.reset_index(drop=True)


        plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
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
    except:
        continue

# %% jupyter={"outputs_hidden": true}
#raw trace visulize
import spikeinterface
import spikeinterface.extractors as se
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'

for i in range(len(Benchmark_pids)):
    try:
        pID = Benchmark_pids[i]
        eID, probe = one.pid2eid(pID)
        rec = se.read_ibl_streaming_recording(
            eID,
            probe + '.ap',
            cache_folder="/local/sicache",
        )
        fs = int(rec.get_sampling_frequency())

        ###

        fig, axs = plt.subplots(1, 2, figsize = (20, 10), width_ratios = [9,1])    

    
        ###

        out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
        parent_dir = Path(out_dir)
        cbin_dir = list(parent_dir.glob('destriped_*.cbin'))[0]

        rec_cbin = sf.read_cbin_ibl(Path(out_dir))
        rec = spikeinterface.core.binaryrecordingextractor.BinaryRecordingExtractor(cbin_dir, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
        chunk_olivier = rec.get_traces()

        ###

        im = axs[0].imshow(chunk_olivier[5000:7000,:].T, aspect="auto", vmin = -4, vmax = 4, cmap=mpl.colormaps['RdBu'], origin='lower')

        axs_divider = make_axes_locatable(axs[0])
        caxs = axs_divider.append_axes("right", size="7%", pad="2%")
        cbs = fig.colorbar(im, cax=caxs)

        df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

        br = BrainRegions()

        df_channels = df_channels.reset_index(drop=False)
        df_channels = df_channels[df_channels.pid == pID]
        df_channels = df_channels.reset_index(drop=True)


        plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                               brain_regions=br, display=True, ax=axs[1], label='right')



        plt.savefig(out_dir + '/raw_trace_destripe_compare' + pID +'.png')
        
    except:
        continue

# %% jupyter={"outputs_hidden": true}
from spike_ephys import cell_type_feature
from tqdm import tqdm
from scipy import signal
import h5py
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'
for i in range(len(Benchmark_pids)):
    try:
        pID = Benchmark_pids[i]
        eID, probe = one.pid2eid(pID)
        # out_dir = main_dir + '/' + eID + '_' + probe
        out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
        h5_path = out_dir + '/' + 'subtraction.h5'
        batch_size = 10000
        fs = 30000
        # try:
        with h5py.File(h5_path) as h5:
            spike_idx = h5["spike_index"][:]
            geom = h5["geom"][:]
            channel_index = h5["channel_index"][:]
        # except:
        #     continue

        spike_num = len(spike_idx)
        h5 = h5py.File(h5_path)
        batch_n = int(np.floor(spike_num/batch_size))

        # peak_value = np.zeros((spike_num,))
        # ptp_duration = np.zeros((spike_num,))
        # halfpeak_duration = np.zeros((spike_num,))
        # peak_trough_ratio = np.zeros((spike_num,))
        # spatial_spread = np.zeros((spike_num,))

        # spatial_non_threshold = np.zeros((spike_num,))
        # reploarization_slope = np.zeros((spike_num,))
        # recovery_slope = np.zeros((spike_num,))

        velocity_above = np.zeros((spike_num,))
        velocity_below = np.zeros((spike_num,))
        # depolarization_slope = np.zeros((spike_num,))

        for i in tqdm(range(batch_n)):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            waveforms = h5["denoised_waveforms"][start_idx:end_idx]
            spk_idx = spike_idx[start_idx:end_idx, 1]

            N_C_nan_idx = np.where(np.isnan(waveforms))
            waveforms[N_C_nan_idx] = 0

            waveforms = signal.resample(waveforms, 1210, axis = 1)
#             peak_value[start_idx:end_idx] = cell_type_feature.peak_value(waveforms)
#             ptp_duration[start_idx:end_idx] = cell_type_feature.ptp_duration(waveforms)
#             halfpeak_duration[start_idx:end_idx] = cell_type_feature.halfpeak_duration(waveforms)
#             peak_trough_ratio[start_idx:end_idx] = cell_type_feature.peak_trough_ratio(waveforms)
#             spatial_spread[start_idx:end_idx] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, spk_idx)

#             spatial_non_threshold[start_idx:end_idx] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, spk_idx)
#             reploarization_slope[start_idx:end_idx] = cell_type_feature.reploarization_slope(waveforms, fs*10)
#             recovery_slope[start_idx:end_idx] = cell_type_feature.recovery_slope(waveforms, fs*10)

            # depolarization_slope[start_idx:end_idx] = cell_type_feature.depolarization_slope(waveforms, fs*10)
            v_above, v_below = cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, n_workers=64 )
            velocity_above[start_idx:end_idx] = v_above
            velocity_below[start_idx:end_idx] = v_below

        start_idx = batch_n * batch_size
        end_idx = None
        waveforms = h5["denoised_waveforms"][start_idx:end_idx]
        spk_idx = spike_idx[start_idx:end_idx, 1]

        N_C_nan_idx = np.where(np.isnan(waveforms))
        waveforms[N_C_nan_idx] = 0

        waveforms = signal.resample(waveforms, 1210, axis = 1)
        # peak_value[start_idx:end_idx] = cell_type_feature.peak_value(waveforms)
        # ptp_duration[start_idx:end_idx] = cell_type_feature.ptp_duration(waveforms)
        # halfpeak_duration[start_idx:end_idx] = cell_type_feature.halfpeak_duration(waveforms)
        # peak_trough_ratio[start_idx:end_idx] = cell_type_feature.peak_trough_ratio(waveforms)
        # spatial_spread[start_idx:end_idx] = cell_type_feature.spatial_spread(waveforms, geom, channel_index, spk_idx)
        # spatial_non_threshold[start_idx:end_idx] = cell_type_feature.spatial_spread_weighted_dist(waveforms, geom, channel_index, spk_idx)
        # reploarization_slope[start_idx:end_idx] = cell_type_feature.reploarization_slope(waveforms, fs*10)
        # recovery_slope[start_idx:end_idx] = cell_type_feature.recovery_slope(waveforms, fs*10)
        
        # depolarization_slope[start_idx:end_idx] = cell_type_feature.depolarization_slope(waveforms, fs*10)
        v_above, v_below = cell_type_feature.velocity(waveforms, geom, channel_index, fs*10, n_workers=64 )
        velocity_above[start_idx:end_idx] = v_above
        velocity_below[start_idx:end_idx] = v_below

        # np.save(out_dir + '/ptp_duration.npy', ptp_duration)
        # np.save(out_dir + '/halfpeak_duration.npy', halfpeak_duration)
        # np.save(out_dir + '/peak_trough_ratio.npy', peak_trough_ratio)
        # np.save(out_dir + '/spatial_spread.npy', spatial_spread)
        # # np.save(out_dir + '/velocity_th_25.npy', velocity)
        # # np.save(out_dir + '/velocity_ci_th_25.npy', ci_err)
        # np.save(out_dir + '/non_threshold_spatial_spread.npy', spatial_non_threshold)
        # np.save(out_dir + '/recovery_slope.npy', recovery_slope)
        # np.save(out_dir + '/reploarization_slope_window_50.npy', reploarization_slope)
        # np.save(out_dir + '/spatial_spread_th12.npy', spatial_spread)
        
        np.save(out_dir + '/above_soma_velocity.npy', velocity_above)
        np.save(out_dir + '/below_soma_velocity.npy', velocity_below)
        # np.save(out_dir + '/deploarization_slope.npy', deploarization_slope)
        
        h5.close()
    except:
        continue

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
import matplotlib.pyplot as plt
plt.hist(velocity_above, bins = np.arange(-2E-5, 2E-5, 1E-6));

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
plt.hist(velocity_below, bins = np.arange(-2E-5, 2E-5, 1E-6));

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
plt.hist(depolarization_slope, bins = np.arange(-350, 0, 10));

# %%
with h5py.File(h5_path) as h5:
    z_abs1 = h5["localizations"][:, 2]
    x1 = h5["localizations"][:, 0]
    y1 = h5["localizations"][:, 1]
    times1 = h5["spike_index"][:, 0] / 30_000
    maxptps1 = h5["maxptps"][:]

# %%
from brainbox.ephys_plots import plot_brain_regions
from ibllib.atlas import BrainRegions
import pandas as pd
br = BrainRegions()
fig, axs = plt.subplots(1, 4, figsize = (10, 20))
not_nan_idx = (~np.isnan(velocity_above)) & (maxptps1>6)
velocity_above = np.clip(velocity_above, -0.2*1E-5, 0.2*1E-5)
# colors = (velocity_above[not_nan_idx] - np.min(velocity_above[not_nan_idx]))/(np.max(velocity_above[not_nan_idx]) - np.min(velocity_above[not_nan_idx]))

axs[0].scatter(x1[not_nan_idx], z_abs1[not_nan_idx], c = velocity_above[not_nan_idx], s = 0.1, cmap = 'RdBu')
axs[0].set_ylim([20, 3840])

not_nan_idx = (~np.isnan(velocity_below)) & (maxptps1>6)
velocity_below = np.clip(velocity_below, -0.2*1E-5, 0.2*1E-5)
# colors = (velocity_below[not_nan_idx] - np.min(velocity_below[not_nan_idx]))/(np.max(velocity_below[not_nan_idx]) - np.min(velocity_below[not_nan_idx]))
axs[1].scatter(x1[not_nan_idx], z_abs1[not_nan_idx], c = velocity_below[not_nan_idx], s = 0.1, cmap = 'RdBu')
axs[1].set_ylim([20, 3840])

depolarization_slope = np.clip(depolarization_slope, -150, 0)

colors = (depolarization_slope - np.min(depolarization_slope))/(np.max(depolarization_slope) - np.min(depolarization_slope))
axs[2].scatter(x1, z_abs1, c = colors, s = 0.1)
axs[2].set_ylim([20, 3840])

df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

br = BrainRegions()

df_channels = df_channels.reset_index(drop=False)
df_channels = df_channels[df_channels.pid == pID]
df_channels = df_channels.reset_index(drop=True)

                
plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                               brain_regions=br, display=True, ax=axs[3], label='right')


# %%
import os
all_pids = []
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    # out_dir = main_dir + '/' + eID + '_' + probe
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    h5_path = out_dir + '/' + 'spatial_spread_th12.npy'#'subtraction.h5'
    if os. path. exists(h5_path):
        print(pID)
        all_pids.append(pID)

# %%
np.save('picked_pids_230417.npy', all_pids)

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/discarded_pIDs'
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'

pID = 'e55266c7-eb05-47bb-b263-1cc08dc3c00c'
eID, probe = one.pid2eid(pID)

out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

rec_cbin = sf.read_cbin_ibl(Path(out_dir))
rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

sub_h5 = subtract.subtraction(
    rec,
    out_folder=out_dir,
    thresholds=[12, 10, 8, 6, 5],
    n_sec_pca=40,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    save_denoised_waveforms=True,
    n_jobs=14,
    loc_workers=1,
    overwrite=False,
    # n_sec_chunk=args.batchlen,
    #save_cleaned_pca_projs_on_n_channels=5,
    loc_feature=("ptp", "peak"),
)

# %%
from spike_psvae import subtract
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

pID ='e55266c7-eb05-47bb-b263-1cc08dc3c00c'
eID, probe = one.pid2eid(pID)

ibl_treamer = Streamer(pid=pID, one=one, typ=stream_type, cache_folder=cache_folder, remove_cached=remove_cached)

ibl_treamer._download_raw_partial(first_chunk=1000, last_chunk=1179)

cbin_url = ibl_treamer.url_cbin

alyx_base_path = one.eid2path(eID)
relpath= alyx_base_path.relative_to(one.cache_dir)

cbin_parent_dir = cache_folder + '/' + str(relpath) + '/raw_ephys_data/' + probe + '/chunk_002000_to_002179'

cbin_dir = list(Path(cbin_parent_dir).glob('*.cbin'))[0]
meta_dir = list(Path(cbin_parent_dir).glob('*.meta'))[0]
ch_dir = list(Path(cbin_parent_dir).glob('*.ch'))[0]

for line in fileinput.input(meta_dir, inplace=1):
    if 'fileTimeSecs' in line:
        line = 'fileTimeSecs=180\n'
    sys.stdout.write(line)

subprocess.run(["python",
                destripe_py_dir,
                str(cbin_dir),
               ],
               check = True
              )

destriped_cbin_dir = list(Path(cbin_parent_dir).glob('destriped_*.cbin'))[0]
destriped_meta_dir = list(Path(cbin_parent_dir).glob('destriped_*.meta'))[0]

# destriped_cbin_dir = cbin_parent_dir + destriped_cbin_dir.name
# destriped_meta_dir = cbin_parent_dir + destriped_meta_dir.name


out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID + '/'
# !mkdir {out_dir}

# !mv {str(destriped_cbin_dir)} {out_dir}
# !mv {str(destriped_meta_dir)} {out_dir}


save_destriped_ch_dir = out_dir + 'destriped_' + ch_dir.name

# !cp {ch_dir} {save_destriped_ch_dir}

out_dir = save_folder + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
rec_cbin = sf.read_cbin_ibl(Path(out_dir))
destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

sub_h5 = subtract.subtraction(
                rec,
                out_folder=out_dir,
                thresholds=[12, 10, 8, 6, 5],
                n_sec_pca=40,
                save_subtracted_tpca_projs=False,
                save_cleaned_tpca_projs=False,
                save_denoised_tpca_projs=False,
                save_denoised_waveforms=True,
                n_jobs=14,
                loc_workers=1,
                overwrite=False,
                # n_sec_chunk=args.batchlen,
                save_cleaned_pca_projs_on_n_channels=5,
                loc_feature=("ptp", "peak"),
            )

# %%
from spike_psvae import subtract
destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

rec_cbin = sf.read_cbin_ibl(Path(out_dir))
rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()

sub_h5 = subtract.subtraction(
    rec,
    out_folder=out_dir,
    thresholds=[12, 10, 8, 6, 5],
    n_sec_pca=40,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    save_denoised_waveforms=True,
    n_jobs=14,
    loc_workers=1,
    overwrite=False,
    # n_sec_chunk=args.batchlen,
    #save_cleaned_pca_projs_on_n_channels=5,
    loc_feature=("ptp", "peak"),
)

# %%
