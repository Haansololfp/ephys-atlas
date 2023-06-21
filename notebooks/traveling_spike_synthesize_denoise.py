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
import matplotlib.pyplot as plt
from neurowaveforms.model import generate_waveform
import torch

# %%
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'
template_raw_wfs_benchmark = np.load(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy')
template_raw_wfs_benchmark = template_raw_wfs_benchmark.item()

# %%
# np_load_old = np.load
# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
Benchmark_pids = list(template_raw_wfs_benchmark.keys())

# %%
a = np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])[:,:,:]
a = torch.as_tensor(a, device=device, dtype=torch.float)

# %%
a[np.arange(100), : , np.ones(100)] = 1

# %%
(a>3).long().shape

# %%
(a[0]*torch.arange(40)).shape

# %%
# import time
T2 = []
for i in range(1000):
    start = time.time()

    col_idx = np.repeat(np.arange(3)[None,:], 100,  axis=0)
    row_idx = np.repeat(np.arange(100)[None,:], 3,  axis=0)
    c = a[np.reshape(row_idx, -1), : , np.reshape(col_idx, -1)]
    d = np.reshape(c, [100, 3, 121])

    end = time.time()
    T2.append(end - start)

# %%
plt.plot(torch.max(a, axis = 0).values);

# %%
aplt.plot(c[10,:,:].T);

# %%
a = np.reshape(np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])[:,:,:], [100*121,40])
b = np.reshape(a[:,0], [100, 121])
fig, axs = plt.subplots(2, 1, figsize = [8, 8])
# plt.plot(np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])[0,:,:])
axs[0].plot(np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])[10,:,0])
axs[1].plot(b[10,:]);

# %%
wfs_array = []
colliding_array = []
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    units = list(template_raw_wfs_benchmark[pID]['temp'].keys())
    
    for j in range(len(units)):
        template =  np.array(template_raw_wfs_benchmark[pID]['temp'][units[j]])
        ptps = template.ptp(0)
        maxchan = np.nanargmax(ptps)
        
        # wfs =  np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])[50,:,:]
        
        wfs_array.append(template[:, maxchan])
        colliding_array.append(np.array(template_raw_wfs_benchmark[pID]['wfs'][units[j]])[50,:,:])

# %%
wfs_array = np.array(wfs_array)
colliding_array = np.array(colliding_array)

# %%
from neurowaveforms.model import generate_waveform
from ibllib.plots import Density
wav = generate_waveform(wfs_array[7,:]);
bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
offset= 10

plt.plot(wav*5000 + bias*offset + np.roll(colliding_array[600,:,:], 25, axis = 0));

# %%
plt.imshow((wav*5000 + np.roll(colliding_array[600,:,:], 25, axis = 0)).T)

# %%
from spike_psvae import denoise
import h5py

# %%
cbin_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_69c9a415-f7fa-4208-887b-1417c1479b48_probe_probe00_pID_1a276285-8b0e-4cc9-9f0a-a3a002978724'
h5_dir = cbin_dir + '/subtraction.h5'

with h5py.File(h5_dir) as h5:
    geom = h5['geom'][:]
    channel_index = h5['channel_index'][:]

# %%
dn = denoise.SingleChanDenoiser().load().to(device)

# %%
savedir = '/moto/stats/users/hy2562/projects/ephys_atlas/synthesize_traveling_denoise/speed = 1'
v = 1
offset = 6
for i in range(500):
    idx = np.random.choice(len(wfs_array), 2)
    wav = generate_waveform(wfs_array[idx[0],:], vertical_velocity_mps = v)
    jitter = np.random.randint(5, 100)
    synth_wfs = wav*2500 + np.roll(colliding_array[idx[1],:,:], jitter, axis = 0)
    device = None
    
    ptps = wav.ptp(0)
    maxchan = np.nanargmax(ptps)
        
    waveforms = torch.as_tensor(synth_wfs[None, :,:], device=device, dtype=torch.float)
    waveforms = denoise.multichan_phase_shift_denoise(waveforms, geom, channel_index, dn, maxchans = [maxchan])
        
    plt.figure(figsize = (6, 10))
    plt.plot(synth_wfs + bias*offset, c = 'k')
    plt.plot(wav*2500 + bias*offset, c = 'r')
    plt.plot(np.squeeze(waveforms) + bias*offset, c = 'g')
    
    plt.savefig(savedir + '/' + 'denoise_waveform_synthesize_travel_speed_' + str(v) + '_' + str(i) + '.png')
    
    plt.close()

# %% [markdown]
# Test on gpurized phase-shift denoiser

# %%
savedir = '/moto/stats/users/hy2562/projects/ephys_atlas/gpurize_phaseshift_denoiser_results/simulated_traveling_data/speed = 1'
v = 1
offset = 6
device = 'cuda'
ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(channel_index, geom, device)
ci_graph_all_maxCH_uniq = denoise.make_ci_graph_all_maxCH(ci_graph_on_probe, maxCH_neighbor, device)

bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
for i in range(500):
    idx = np.random.choice(len(wfs_array), 2)
    wav = generate_waveform(wfs_array[idx[0],:], vertical_velocity_mps = v)
    jitter = np.random.randint(5, 100)
    synth_wfs = wav*2500 + np.roll(colliding_array[idx[1],:,:], jitter, axis = 0)
    # device = None
    
    ptps = wav.ptp(0)
    maxchan = np.nanargmax(ptps)
        
    waveforms = torch.as_tensor(synth_wfs[None, :,:], device=device, dtype=torch.float)
    
    maxchans = torch.tensor([maxchan], device=device)
    
    waveforms, wfs_old_denoiser = denoise.multichan_phase_shift_denoise_preshift(waveforms, ci_graph_all_maxCH_uniq, maxCH_neighbor, dn, maxchans, device)
    
    waveforms = waveforms.cpu().detach().numpy()
    plt.figure(figsize = (6, 10))
    plt.plot(synth_wfs + bias*offset, c = 'k')
    plt.plot(wav*2500 + bias*offset, c = 'r')
    plt.plot(np.squeeze(waveforms) + bias*offset, c = 'g')
    
    plt.savefig(savedir + '/' + 'denoise_waveform_synthesize_travel_speed_' + str(v) + '_' + str(i) + '.png')
    
    plt.close()

# %%
maxchans

# %%
