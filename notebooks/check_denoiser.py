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
# check out whether the denoiser screw up the waveform shape

# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import spikeinterface.full as sf

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/eID_b03fbc44-3d8e-4a6c-8a50-5ea3498568e0_probe_probe00_pID_9117969a-3f0d-478b-ad75-98263e3bfacf'

# %%
h5_File = main_dir + '/subtraction.h5'
with h5py.File(h5_File) as h5:
    denoised_wfs = h5["denoised_waveforms"][:]
    spike_index = h5["spike_index"][:]
    channel_index = h5["channel_index"][:]
    geom = h5["geom"][:]

# %%
spk_channels = spike_index[:,1]
which = np.squeeze(np.where((spk_channels>200) & (spk_channels<350)))

# %%
spk_times = spike_index[:,0]

# %%
parent_dir = Path(main_dir)
destriped_cbin_dir = list(parent_dir.glob('destriped_*.cbin'))[0]

# %%
destriped_cbin = main_dir + '/' + destriped_cbin_dir.name

# %%
idx = 20

# %%
rec_cbin = sf.read_cbin_ibl(Path(main_dir))
recording = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
load_start = spk_times - 42
load_end = spk_times + 79

ci = channel_index[spk_channels[idx]]
raw_wfs = recording.get_traces(start_frame=load_start[idx], end_frame=load_end[idx])[:,ci]

bias = np.arange(len(ci))
bias = np.repeat(bias[None,:], 121, axis = 0)

plt.figure(figsize = (4, 8))

plt.plot(raw_wfs + bias * 5, 'k');#, aspect = 'auto')

plt.plot(denoised_wfs[idx] + bias * 5, '--r');

# %% jupyter={"outputs_hidden": true}
fig, axs = plt.subplots(25, 4, figsize = (16, 200))
for i in range(100):
    idx = which[i + 1000]
    ci = channel_index[spk_channels[idx]]
    raw_wfs = recording.get_traces(start_frame=load_start[idx], end_frame=load_end[idx])[:,ci]

    bias = np.arange(len(ci))
    bias = np.repeat(bias[None,:], 121, axis = 0)
    
    row = i//4
    col = np.mod(i, 4)
    

    axs[row][col].plot(raw_wfs + bias * 5, 'k');#, aspect = 'auto')
    
    other_spk_times = spk_times[(spk_times>load_start[idx]) & (spk_times<load_end[idx]) & (spk_channels>spk_channels[idx] - 20) & (spk_channels<spk_channels[idx] + 20)]
    
    axs[row][col].vlines(other_spk_times - load_start[idx], -10, 210);
    axs[row][col].vlines(42, -10, 210 , 'y');
    
    axs[row][col].plot(denoised_wfs[idx] + bias * 5, 'r');
    
    axs[row][col].set_title('row:' + str(row) + ', col:' + str(col) + ', unit:' + str(idx))
    
plt.savefig('denoiser_compare_raw_2.png')

# %%
above_velocity_dir = main_dir + '/above_soma_velocity.npy'
above_velocity = np.load(above_velocity_dir)

# %%
check_PIDs = dict()
check_PIDs['9117969a-3f0d-478b-ad75-98263e3bfacf'] = 3001000
check_PIDs['80f6ffdd-f692-450f-ab19-cd6d45bfd73e'] = 601000
check_PIDs['3eb6e6e0-8a57-49d6-b7c9-f39d5834e682'] = 4801000
check_PIDs['ad714133-1e03-4d3a-8427-33fc483daf1a'] = 3601000

# %%
from one.api import ONE
from pathlib import Path
import spikeinterface.full as sf

main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'

for i in range(4):
    one = ONE(base_url="https://alyx.internationalbrainlab.org")

    pID = list(check_PIDs.keys())[i]
    eID, probe = one.pid2eid(pID)
    
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID
    h5_path = out_dir + '/' + 'subtraction.h5'

    sample_start = check_PIDs[pID]

    destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()

    rec_chunk = rec.frame_slice(start_frame=sample_start,end_frame=sample_start + 2000)
    
    with h5py.File(h5_path) as h5:
        spike_index = h5["spike_index"][:]
    
    t_samples = spike_index[:, 0]

    in_chunk = (t_samples >= sample_start) & (t_samples < sample_start + 2000)
    spike_index_chunk = spike_index[in_chunk].copy()
    spike_index_chunk[:, 0] -= sample_start
    
    data = rec_chunk.get_traces()
    
    
    np.save('raw_traces_to_check_pID_' + pID + '_Ts_' + str(sample_start), data)
    np.save('spike_index_in_chunk_pID' + pID + '_Ts_' + str(sample_start), spike_index_chunk)

# %%
import scipy
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets/visualize_feature_value_on_raw'
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_similar_datasets'

for i in range(4):
    pID = list(check_PIDs.keys())[i]
    eID, probe = one.pid2eid(pID)
    
    sample_start = check_PIDs[pID]
    
    spike_pick_mat = save_dir + '/pID' + pID + '_Ts_' + str(sample_start) + '_spike_pick.mat'
    spike_pick = scipy.io.loadmat(spike_pick_mat)
    
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        spike_index = h5["spike_index"][:]
        channel_index = h5["channel_index"][:]
        
    spk_times = spike_index[:,0]
    spk_channels = spike_index[:,1]
    
    destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    load_start = spk_times - 42
    load_end = spk_times + 79
    
    spk_idx = []
    for j in range(len(spike_pick['curve_travel_spike'][0])):
        # print(spike_pick['curve_travel_spike'][0][j][1][0][0])
        spk_idx.append(np.where(spike_index[:,0]==spike_pick['curve_travel_spike'][0][j][1][0][0] + sample_start)[0][0])
        
        
    spk_idx = np.squeeze(np.array(spk_idx))
    
    with h5py.File(h5_path) as h5:
        denoised_wfs = h5["denoised_waveforms"][np.sort(spk_idx),:,:]
    
    for j in range(len(spk_idx)):
        fig, axs = plt.subplots(1, 3, figsize = (12, 8))
        axs[0].imshow(denoised_wfs[j,:,:].T, vmin = -3, vmax = 3, aspect = 'auto', cmap = 'RdBu', origin = 'lower')
        axs[0].set_title('denoised_waveform')
        
        idx = np.sort(spk_idx)[j]
        ci = channel_index[spk_channels[idx]]
        raw_wfs = rec.get_traces(start_frame=load_start[idx], end_frame=load_end[idx])[:,ci]
        
        axs[1].imshow(raw_wfs.T, vmin = -3, vmax = 3, aspect = 'auto', cmap = 'RdBu', origin = 'lower')
        axs[1].set_title('raw_waveform')
        
        bias = np.arange(len(ci))
        bias = np.repeat(bias[None,:], 121, axis = 0)
        
        
        axs[2].plot(raw_wfs + bias * 8, 'k')
        axs[2].plot(denoised_wfs[j,:,:] + bias * 8, 'r')
        axs[2].set_title('traces_on_top')
        
        plt.savefig(save_dir + '/curvy_trace_denoiser/pID_' + pID + '_curvy_spike_denoise_T_' + str(spk_times[idx]) + '.png')

# %%
t = np.arange(38, 58, 1)

mini_patch = np.array([[0, 0], [0, 2], [1, 1], [1,3]])
for i in range(4):
    pID = list(check_PIDs.keys())[i]
    eID, probe = one.pid2eid(pID)
    
    sample_start = check_PIDs[pID]
    
    spike_pick_mat = save_dir + '/pID' + pID + '_Ts_' + str(sample_start) + '_spike_pick.mat'
    spike_pick = scipy.io.loadmat(spike_pick_mat)
    
    out_dir = main_dir + '/eID_' + eID + '_probe_' + probe + '_pID_' + pID

    h5_path = out_dir + '/' + 'subtraction.h5'
    with h5py.File(h5_path) as h5:
        spike_index = h5["spike_index"][:]
        channel_index = h5["channel_index"][:]
        
    spk_times = spike_index[:,0]
    spk_channels = spike_index[:,1]
    
    destriped_cbin_dir = list(Path(out_dir).glob('destriped_*.cbin'))[0]
    destriped_cbin = out_dir + '/' + destriped_cbin_dir.name

    rec_cbin = sf.read_cbin_ibl(Path(out_dir))
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    load_start = spk_times - 42
    load_end = spk_times + 79
    
    spk_idx = []
    for j in range(len(spike_pick['curve_travel_spike'][0])):
        # print(spike_pick['curve_travel_spike'][0][j][1][0][0])
        spk_idx.append(np.where(spike_index[:,0]==spike_pick['curve_travel_spike'][0][j][1][0][0] + sample_start)[0][0])

    spk_idx = np.squeeze(np.array(spk_idx))

    with h5py.File(h5_path) as h5:
        denoised_wfs = h5["denoised_waveforms"][np.sort(spk_idx),:,:]

    for j in range(len(spk_idx)):
        fig, axs = plt.subplots(1, 20, figsize = (16, 4))
        idx = spk_idx[j]
        ci = channel_index[spk_channels[idx]]
        
        for k in range(20):
            probe = np.zeros((192, 4)) * np.nan
            patch_n = ci // 4
            ind = np.mod(ci, 4)

            snap_shot = denoised_wfs[j,t[k],:]

            row = patch_n*2 + mini_patch[ind][:,0]
            col = mini_patch[ind][:,1]

            probe[row, col] = snap_shot
            axs[k].imshow(probe[np.min(row):np.max(row) + 1,:], cmap = 'RdBu', vmin = -7, vmax = 7, aspect = 'auto')
            axs[k].set_yticklabels([])

# %%
