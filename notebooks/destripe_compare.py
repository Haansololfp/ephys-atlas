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
import spikeinterface
import spikeinterface.preprocessing as si
import spikeinterface.extractors as se
from pathlib import Path
import matplotlib.pyplot as plt

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

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'

for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    rec = se.read_ibl_streaming_recording(
        eID,
        probe + '.ap',
        cache_folder="/local/sicache",
    )
    fs = int(rec.get_sampling_frequency())
    
    ###
    
    fig, axs = plt.subplots(3, 1, figsize = (20, 40))    

    rec = rec.frame_slice(start_frame=int(2000*fs),end_frame=int(2180*fs))
    chunk_raw = rec.get_traces()
    
    ###
    # chunk = rec.get_traces()
    im = axs[0].imshow(chunk_raw[1000:3000,:].T, aspect="auto")
    axs_divider = make_axes_locatable(axs[0])
    caxs = axs_divider.append_axes("right", size="7%", pad="2%")
    cbs = fig.colorbar(im, cax=caxs)
    axs[0].set_title('raw_traces')
    # axs[0].set_xlim([0, 1000])
    ###
    
    rec = si.highpass_filter(rec)
    rec = si.phase_shift(rec)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec, num_random_chunks=100)
    print(f"{bad_channel_ids=}")
    
    rec = rec.frame_slice(start_frame=0,end_frame=3000000)
    rec = si.interpolate_bad_channels(rec, bad_channel_ids)
    rec = si.highpass_spatial_filter(rec)
    # we had been working with this before -- should switch to MAD,
    # but we need to rethink the thresholds
    
    rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100)
    
    ###
    chunk_si = rec.get_traces()
    im2 = axs[1].imshow(chunk_si[1000:3000].T, aspect="auto", vmin = -20, vmax = 6)
    axs2_divider = make_axes_locatable(axs[1])
    caxs2 = axs2_divider.append_axes("right", size="7%", pad="2%")
    cbs2 = fig.colorbar(im2, cax=caxs2)
    axs[1].set_title('spike_interface_traces')
    # axs[1].set_xlim([0, 1000])
    
    ###
    
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    parent_dir = Path(out_dir)
    cbin_dir = list(parent_dir.glob('destriped_*.cbin'))[0]
    
    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    rec = spikeinterface.core.binaryrecordingextractor.BinaryRecordingExtractor(cbin_dir, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    chunk_olivier = rec.get_traces()

    ###
    
    im3 = axs[2].imshow(chunk_olivier[1000:3000,:].T, aspect="auto", vmin = -20, vmax = 6)
    axs3_divider = make_axes_locatable(axs[2])
    caxs3 = axs3_divider.append_axes("right", size="7%", pad="2%")
    cbs3 = fig.colorbar(im3, cax=caxs3)
    
    axs[2].set_title('olivier destripe')
    # axs[2].set_xlim([0, 1000])
    
    plt.savefig(out_dir + '/raw_trace_destripe_compare' + pID +'.png')

# %%
import spikeinterface.full as sf
fig, axs = plt.subplots(3, 1, figsize = (20, 40))

###
# chunk = rec.get_traces()
im = axs[0].imshow(chunk_raw[1000:3000,:].T, aspect="auto")
axs_divider = make_axes_locatable(axs[0])
caxs = axs_divider.append_axes("right", size="7%", pad="2%")
cbs = fig.colorbar(im, cax=caxs)
axs[0].set_title('raw_traces')
# axs[0].set_xlim([0, 1000])

###

im2 = axs[1].imshow(chunk_si[1000:3000,:].T, aspect="auto", vmin = -20, vmax = 6)
axs2_divider = make_axes_locatable(axs[1])
caxs2 = axs2_divider.append_axes("right", size="7%", pad="2%")
cbs2 = fig.colorbar(im2, cax=caxs2)
axs[1].set_title('spike_interface_traces')
# axs[1].set_xlim([0, 1000])
# axs[1].set_clim([-20,6])
###

# out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
# parent_dir = Path(out_dir)
# cbin_dir = list(parent_dir.glob('destriped_*.cbin'))[0]

# rec_cbin = sf.read_cbin_ibl(Path(out_dir))
# rec = spikeinterface.core.binaryrecordingextractor.BinaryRecordingExtractor(cbin_dir, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
# chunk_olivier = rec.get_traces()


###

im3 = axs[2].imshow(chunk_olivier[1000:3000,:].T, aspect="auto", vmin = -20, vmax = 6)
axs3_divider = make_axes_locatable(axs[2])
caxs3 = axs3_divider.append_axes("right", size="7%", pad="2%")
cbs3 = fig.colorbar(im3, cax=caxs3)

axs[2].set_title('olivier destripe')
# axs[2].set_xlim([0, 1000])
# axs[2].set_clim([-20,6])

# %%
plt.imshow(chunk[0:1000,:].T, aspect="auto")
plt.clim([-20,6])
plt.cmap

# %%
import numpy as np
np.shape(chunk)

# %%
np.min(chunk[0:1000,:])

# %%
np.max(chunk[0:1000,:])

# %%
