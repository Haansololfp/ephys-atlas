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
# Test the idea of two channel denoising

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from spike_ephys import waveform_denoising_two_channels

import numpy as np
from spike_ephys import waveform_noise
from one.api import ONE
import spikeinterface.full as sf
from pathlib import Path
import h5py
one = ONE(base_url="https://alyx.internationalbrainlab.org")

# %%
template_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark/manual_picked_temp.npy'
templates = np.load(template_dir)

# %%
ptps = templates.ptp(1)
mc = np.argmax(ptps, axis =1)

# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'

# %%
pID = 'dab512bd-a02d-4c1f-8dbc-9155a163efc0'
eID, probe = one.pid2eid(pID)
check_dir = main_dir + '/' + 'eID_' + eID + '_probe_' + probe + '_pID_' + pID

destriped_cbin_dir = list(Path(check_dir).glob('destriped_*.cbin'))[0]
    
rec_cbin = sf.read_cbin_ibl(Path(check_dir))
destriped_cbin = check_dir + '/' + destriped_cbin_dir.name
#rec_cbin.get_num_channels()
rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
rec.set_probe(rec_cbin.get_probe(), in_place=True)
fs = rec.get_sampling_frequency()


recordings = rec.get_traces()


spatial_SIG, temporal_SIG = waveform_noise.noise_whitener(recordings, temporal_size = 121, window_size = 121, sample_size=1000,
                   threshold=4.0, max_trials_per_sample=1000,
                   allow_smaller_sample_size=False)

# %%
h5_dir = check_dir + '/subtraction.h5'
with h5py.File(h5_dir) as h5:
    channel_index = h5['channel_index'][:]
    geom = h5['geom'][:]

# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,2, figsize = (20, 10))
axs[0].imshow(spatial_SIG)
axs[1].imshow(temporal_SIG)

# %%
plt.plot(np.squeeze(waveform_denoising_two_channels.make_noise(1, spatial_SIG, temporal_SIG))[:,0:15]);

# %%
two_CH_DN = waveform_denoising_two_channels.TwoChanDenoiser()

# %%
DenoTD = waveform_denoising_two_channels.Denoising_Training_Data_Two_Channels(templates,
                                                                              mc,
                                                                              channel_index,
                                                                              spatial_SIG,
                                                                              temporal_SIG,
                                                                              geom)

# %%
fname_save = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark/two_chan_denoiser_ptp_th_2.pt'

# %%
n_test = 500

# %%
wf_col_test, wf_clean_test = DenoTD.make_training_data(n_test)

# %%
from spike_psvae.denoise import SingleChanDenoiser
dn = SingleChanDenoiser().load()
wfs = wf_col_test
wfs_denoised_old = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
wfs_denoised_old = wfs_denoised_old.detach().numpy()

# %%
plt.plot(wfs_denoised_old[i,1,:])

# %%
denoised = two_CH_DN.forward(torch.FloatTensor(wf_col_test))
denoised = denoised.detach().numpy()
fig, axs = plt.subplots(10, 10, figsize = (60, 60))
for i in range(100):
    row = i//10
    col = np.mod(i, 10)
    axs[row][col].plot(wf_clean_test.transpose(0,2,1)[i,:,1])
    axs[row][col].plot(wf_col_test.transpose(0,2,1)[i,:,1])
    axs[row][col].plot(denoised[i,:])
    axs[row][col].plot(wfs_denoised_old[i,1,:])
    # axs[row][col].legend(['clean', 'noisy', 'denoised'])
    
plt.savefig('/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark/test_set_denoise_ptp_2.png')

# %%
plt.plot(denoised[1,:])

# %%

# %%
TwoChanDN = waveform_denoising_two_channels.TwoChanDenoiser(pretrained_path = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark/two_chan_denoiser.pt')
TwoChanDN = TwoChanDN.load()
SingleChanDN = SingleChanDenoiser().load()

iterative_denoise(wf, TwoChanDN, SingleChanDN, max_CH, geom, channel_index)

# %%
1

# %%
np.shape(np.linalg.norm(DenoTD.two_channel_templates[:,:,0], ord = 2, axis = 1))

# %%
import scipy
CH_wf_dist = np.sum(DenoTD.two_channel_templates[:,:,0] * DenoTD.two_channel_templates[:,:,1], axis = 1)/(np.linalg.norm(DenoTD.two_channel_templates[:,:,0], ord = 2, axis = 1)*np.linalg.norm(DenoTD.two_channel_templates[:,:,1], ord = 2, axis = 1))


# %%
CH_wf_dist

# %%
plt.hist(CH_wf_dist)

# %%
two_CH_DN.train(fname_save, DenoTD)

# %%

# %%
