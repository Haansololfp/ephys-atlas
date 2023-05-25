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
from spike_psvae import denoise
import h5py
import torch

# %%
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# %%
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'
template_raw_wfs_benchmark = np.load(manually_picked_temp_dir + '/templates_w_raw_waveforms.npy')

# %%
template_raw_wfs_benchmark = template_raw_wfs_benchmark.item()

# %%
wfs_to_denoise = np.array(template_raw_wfs_benchmark['1a276285-8b0e-4cc9-9f0a-a3a002978724']['wfs'][10])

# %%
np.sort(np.array(list(template_raw_wfs_benchmark['1a276285-8b0e-4cc9-9f0a-a3a002978724']['wfs'].keys())))

# %%
np.shape(wfs_to_denoise)

# %%
a = np.reshape(np.swapaxes(wfs_to_denoise, 1, 2), [100*40, 121])

# %%
bias = bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
offset= 10
i = 80
idx = np.arange(0, 40) + i*40
plt.plot(a[idx,:].T + bias*offset);
plt.plot(wfs_to_denoise[i,:,:]+ bias*offset);

# %% jupyter={"outputs_hidden": true}
plt.figure(figsize = [6, 100])
plt.imshow(a, aspect = 'auto', vmin = -3, vmax = 3)

# %%
# import torch
device = None
waveforms = torch.as_tensor(wfs_to_denoise, device=device, dtype=torch.float)

# %%
import h5py

# %%
h5_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract/eID_69c9a415-f7fa-4208-887b-1417c1479b48_probe_probe00_pID_1a276285-8b0e-4cc9-9f0a-a3a002978724'
h5_dir = h5_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    channel_index = h5['channel_index'][:]
    geom = h5['geom'][:]

# %%
print(channel_index[378])

# %%
plt.plot(maxCH_neighbor.T);

# %%
from spike_psvae.denoise import SingleChanDenoiser

ci_graph_on_probe, maxCH_neighbor = denoise.make_ci_graph(channel_index, geom)
dn = SingleChanDenoiser().load()
denoised_waveforms = denoise.multichan_phase_shift_denoise(waveforms, ci_graph_on_probe, torch.tensor(maxCH_neighbor).type(torch.LongTensor),  dn, np.ones(100)*304)

# %%
ci_graph_on_probe[380]

# %%
b = dict()
b[0] = [1,9, 0]

b[2] = [1]


a = np.arange(21)
a = np.reshape(a, [3, 7])

v = ci_graph_on_probe[0][0]
b[3] = list(v[a[2,v] == 3])
b[4] = list(v[a[0,v] == 3])
b[1] = [2,3]

# %%
from itertools import zip_longest
import numpy as np
for nodes in zip_longest(*dict(sorted(b.items())).values()):
    print(nodes)
    print(np.where(np.array(nodes)!=None)[0])

# %%
ci_graph_on_probe[0][0]

# %%
waveforms = waveforms.detach().numpy()
denoised_waveforms = denoised_waveforms.detach().numpy()

# %%
bias = bias = np.arange(40)
bias = np.repeat(bias[None,:], 121, axis = 0)
offset= 10
i = 0

plt.plot(waveforms[i,:, :] + bias*offset, c = 'k');
plt.plot(denoised_waveforms[i,:,:] + bias*offset, c = 'r' );

# %%
maxCH_neighbor

# %% jupyter={"outputs_hidden": true}
ci_graph_on_probe

# %%
