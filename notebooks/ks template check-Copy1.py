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
from one.api import ONE
import spikeinterface.preprocessing as si
import spikeinterface.extractors as se
import spikeinterface.full as sf
from spike_psvae import subtract
from pathlib import Path
import subprocess
import fileinput
import numpy as np
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas

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
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/improved_destripe/destripe_and_subtract'

# %%
one = ONE(base_url="https://alyx.internationalbrainlab.org")

# %%
ba = AllenAtlas()

# %%
channel_index_all = np.arange(384)
channel_index_all = np.tile(channel_index_all,(384,1))

# %% jupyter={"outputs_hidden": true}
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/'

for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    check_dir = main_dir + '/' + 'eID_' + eID + '_probe_' + probe + '_pID_' + pID
    # for j in range(len(snippet_path)):
    destriped_cbin_dir = list(Path(check_dir).glob('destriped_*.cbin'))[0]
    
    rec_cbin = sf.read_cbin_ibl(Path(check_dir))
    destriped_cbin = check_dir + '/' + destriped_cbin_dir.name
    #rec_cbin.get_num_channels()
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    
    sl = SpikeSortingLoader(eid=eID, pname=probe,  one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    
    good_clusters = np.squeeze(np.where(clusters['metrics']["label"] == 1))
    st = sl.samples2times((spikes["times"]), direction='reverse')
    
    h5_dir = check_dir + '/subtraction.h5'
    with h5py.File(h5_dir) as h5:
        channel_index = h5["channel_index"][:]
        geom = h5["geom"][:]
    
    cluster_wfs = dict()
    
    max_channels = []
    
    for j in range(len(good_clusters)):
        
        spks_in_cluster = np.squeeze(np.where((spikes["clusters"] == good_clusters[j]) & (st > sl.samples2times(2000, direction='reverse') + 42) & (st < sl.samples2times(2180, direction='reverse')  - 79)))
        
        if np.shape(spks_in_cluster):
            if len(spks_in_cluster)>300:
                cluster_wfs[good_clusters[j]] = []
                c = []
                for k in range(len(spks_in_cluster)): 

                    spk_channel_depth = spikes['depths'][spks_in_cluster[k]]
                    maxCH = np.argmin(np.abs(geom[:,1] - spk_channel_depth))
                    c.append(maxCH)

                    ci = channel_index[maxCH]

                    load_start = np.int32(st[spks_in_cluster[k]] - 42 - sl.samples2times(2000, direction='reverse'))
                    load_end = np.int32(st[spks_in_cluster[k]] + 79 - sl.samples2times(2000, direction='reverse'))

                    raw_wfs = np.zeros((121, 40)) * np.nan
                    ci_not_nan = np.squeeze(np.where(ci < 384))


                    raw_wfs[:,ci_not_nan] = rec.get_traces(start_frame=load_start, end_frame=load_end)[:,ci[ci_not_nan]]


                    cluster_wfs[good_clusters[j]].append(raw_wfs)
                c = np.array(c)
                bcount = np.bincount(c) 
                max_channels.append(np.argmax(bcount))
                
    good_clusters = list(cluster_wfs.keys())
    corrected_maxCHs = []
    
    for j in range(len(good_clusters)):
        max_channel = max_channels[j]
        fig = plt.figure(figsize = (10,30))
        picked_wfs = np.array(cluster_wfs[good_clusters[j]])[0:100,:,:]
        max_abs_amp = np.max(np.abs(picked_wfs))

        lines = cluster_viz_index.pgeom(picked_wfs, max_channel*np.ones((100,), int), channel_index=channel_index, geom=geom, color = [0.5, 0.5, 0.5], max_abs_amp = max_abs_amp);

        lines = cluster_viz_index.pgeom(np.nanmean(picked_wfs, axis = 0)[None,:,:], max_channel, channel_index=channel_index, geom=geom, color = 'r', max_abs_amp = max_abs_amp, show_chan_label=True);

        ptps = np.ptp(np.nanmean(picked_wfs, axis = 0), axis = 0)
        corrected_maxCH = channel_index[max_channel][np.argmax(ptps)]
        corrected_maxCHs.append(corrected_maxCH)
        title = pID + '_unit_' + str(good_clusters[j]) + '_maxCH_' + str(corrected_maxCH)
        plt.title(title)
        plt.savefig(save_dir + title)
        plt.close()
        
    corrected_maxCHs = np.array(corrected_maxCHs)
    np.savez(check_dir + '/ks_cluster_waveforms.npz', cluster_wfs = cluster_wfs, max_channels = max_channels, corrected_maxCHs = corrected_maxCHs)

# %%
# from spike_psvae import cluster_viz_index
good_clusters = list(cluster_wfs.keys())
i = 15

# fig1, axs = plt.subplots(8, 4, figsize = (10, 60))
with h5py.File(h5_dir) as h5:
    geom = h5["geom"][:]
max_channel = max_channels[i]
fig = plt.figure(figsize = (10,30))
picked_wfs = np.array(cluster_wfs[good_clusters[i]])[0:100,:,:]
max_abs_amp = np.max(np.abs(picked_wfs))
wfs_with_mean = np.append(picked_wfs, np.nanmean(picked_wfs, axis = 0)[None,:,:], axis = 0)

lines = cluster_viz_index.pgeom(picked_wfs, max_channel*np.ones((100,), int), channel_index=channel_index, geom=geom, color = [0.5, 0.5, 0.5], max_abs_amp = max_abs_amp);

lines = cluster_viz_index.pgeom(np.nanmean(picked_wfs, axis = 0)[None,:,:], max_channel, channel_index=channel_index, geom=geom, color = 'r', max_abs_amp = max_abs_amp, show_chan_label=True);


ptps = np.ptp(np.nanmean(picked_wfs, axis = 0), axis = 0)
corrected_maxCH = channel_index[max_channel][np.argmax(ptps)]

plt.title(pID + '_unit_' + str(good_clusters[i]) + '_maxCH_' + str(corrected_maxCH))


# plt.close()


# %%
2000*fs

# %%
from spike_psvae.spike_train_utils import clean_align_and_get_templates
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/'

for i in [5, 6, 7]:#range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    check_dir = main_dir + '/' + 'eID_' + eID + '_probe_' + probe + '_pID_' + pID
    # for j in range(len(snippet_path)):
    destriped_cbin_dir = list(Path(check_dir).glob('destriped_*.cbin'))[0]
    
    rec_cbin = sf.read_cbin_ibl(Path(check_dir))
    destriped_cbin = check_dir + '/' + destriped_cbin_dir.name
    #rec_cbin.get_num_channels()
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    
    sl = SpikeSortingLoader(eid=eID, pname=probe,  one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    
    good_clusters = np.squeeze(np.where(clusters['metrics']["label"] == 1))
    st = sl.samples2times((spikes["times"]), direction='reverse')


    spks_in_cluster = np.squeeze(np.where((np.in1d(spikes["clusters"], good_clusters)) & (st > (2000*30000 + 42)) & (st <  (2180*30000  - 79))))

    spike_train = np.concatenate([st[spks_in_cluster][:, None] -  2000*30000 , spikes["clusters"][spks_in_cluster][:, None]], axis = 1)

    (
        aligned_spike_train,
        order,
        templates_aligned,
        template_shifts,
    ) = clean_align_and_get_templates(
        np.int32(spike_train),
        384,
        destriped_cbin,
        min_n_spikes=300,
    )
    
    np.savez(check_dir + '/ks_cluster_templates.npz', aligned_spike_train = aligned_spike_train, order = order, templates_aligned = templates_aligned, template_shifts = template_shifts)

# %%
template = templates_aligned[good_clusters[j],:,:]

# %% jupyter={"outputs_hidden": true}
save_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/template_from_benchmark/'

for i in [5, 6, 7]:#range(11, 12):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    check_dir = main_dir + '/' + 'eID_' + eID + '_probe_' + probe + '_pID_' + pID
    # for j in range(len(snippet_path)):
    destriped_cbin_dir = list(Path(check_dir).glob('destriped_*.cbin'))[0]
    
    rec_cbin = sf.read_cbin_ibl(Path(check_dir))
    destriped_cbin = check_dir + '/' + destriped_cbin_dir.name
    #rec_cbin.get_num_channels()
    rec = sf.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
    rec.set_probe(rec_cbin.get_probe(), in_place=True)
    fs = rec.get_sampling_frequency()
    
    ks_cluster_templates_dir = check_dir + '/ks_cluster_templates.npz'
    
    ks_cluster_templates = np.load(ks_cluster_templates_dir)
    
    aligned_spike_train = ks_cluster_templates['aligned_spike_train']
    order = ks_cluster_templates['order']
    templates_aligned = ks_cluster_templates['templates_aligned']
    
    st = aligned_spike_train[order,:]
    cluster_ids = np.unique(aligned_spike_train[:,1])
    
    
    cluster_wfs = dict()
    max_channels = []
    
    for j in range(1, len(cluster_ids)):
        
        template = templates_aligned[j - 1,:,:]
        ptps = template.ptp(0)
        mc = np.argmax(ptps)
        max_channels.append(mc)
        
        ci = channel_index[mc]
        
        clu_id = cluster_ids[j]
        
        cluster_wfs[clu_id] = []
        
        spks_in_cluster = np.squeeze(np.where(st[:,1] == clu_id))
        rand_sample_idx = np.random.choice(len(spks_in_cluster), 100)
        
        for k in range(100):
            load_start = np.int32(st[spks_in_cluster][rand_sample_idx[k], 0] - 42)
            load_end = np.int32(st[spks_in_cluster][rand_sample_idx[k], 0] + 79)

            raw_wfs = np.zeros((121, 40)) * np.nan
            ci_not_nan = np.squeeze(np.where(ci < 384))


            raw_wfs[:,ci_not_nan] = rec.get_traces(start_frame=load_start, end_frame=load_end)[:,ci[ci_not_nan]]
            
            cluster_wfs[clu_id].append(raw_wfs)
      
    good_clusters = list(cluster_wfs.keys())
    
    h5_dir = check_dir + '/subtraction.h5'
    with h5py.File(h5_dir) as h5:
        geom = h5["geom"][:]
        
    for j in range(len(good_clusters)):
        max_channel = max_channels[j]
        fig = plt.figure(figsize = (10,30))
        picked_wfs = np.array(cluster_wfs[good_clusters[j]])[0:100,:,:]
        max_abs_amp = np.max(np.abs(picked_wfs))
        
        ci = channel_index[max_channel]
        template = templates_aligned[good_clusters[j],:,:]
        template_wfs = np.zeros((121, 40)) * np.nan
        not_nan_chan = np.squeeze(np.where(ci <384))
        template_wfs[:, not_nan_chan] = template[:, ci[not_nan_chan]]
        
        lines = cluster_viz_index.pgeom(picked_wfs, max_channel*np.ones((100,), int), channel_index=channel_index, geom=geom, color = [0.5, 0.5, 0.5], max_abs_amp = max_abs_amp);
        
        lines = cluster_viz_index.pgeom(template_wfs, max_channel, channel_index=channel_index, geom=geom, color = 'r', max_abs_amp = max_abs_amp, show_chan_label=True);


        ptps = np.ptp(np.nanmean(picked_wfs, axis = 0), axis = 0)
        corrected_maxCH = channel_index[max_channel][np.argmax(ptps)]

        title = pID + '_unit_' + str(good_clusters[j]) + '_maxCH_' + str(corrected_maxCH)
        plt.title(title)
        plt.savefig(save_dir + title)
        plt.close()


# %%
temp = template[:,ci]


# %%
two_channel_templates = []
x_pitch = np.diff(np.unique(geom[:,0]))[0]
y_pitch = np.diff(np.unique(geom[:,1]))[0]

wfs_ptp = temp.ptp(0)
n_channels = 40
# create a graph with all the neighboring channels connected
maxCH = max_channel #max channel [0, 384]
ci = channel_index[maxCH]
ci_graph = dict()
ci_geom = geom[ci]
for ch in range(len(ci)):
    ci_graph[ch] = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == y_pitch))|
                       ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * y_pitch)) |
                       ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0)))  

mcs_idx = np.squeeze(np.where(ci == maxCH))
CH_checked = np.zeros(n_channels)

CH_checked[mcs_idx] = 1
q = []
q.append(int(mcs_idx))
parents = np.zeros(n_channels, int)

parents[mcs_idx] = 0

full_temp = temp

while len(q)>0:
    u = q.pop()
    v = ci_graph[u][0]
    # v = np.random.shuffle(v) # randomly shuffle, is it necessary?
    for k in v:
        if CH_checked[k] == 0:
            neighbors = ci_graph[k][0]
            checked_neighbors = neighbors[CH_checked[neighbors] == 1]

            CH_ref = np.argmax(wfs_ptp[checked_neighbors])

            parents[k] = checked_neighbors[CH_ref]

            q.insert(0,k)
            CH_checked[k] = 1
            
            if (wfs_ptp[k] > 3) & (wfs_ptp[parents[k]] > 3):
                two_channel_templates.append(full_temp[:, [parents[k], k]])

two_channel_templates = np.array(two_channel_templates)


# %%
np.shape(two_channel_templates)

# %%
plt.plot(two_channel_templates[6,:,:])

# %% jupyter={"outputs_hidden": true}
n_times = 121
up_factor = 5

np.arange(0, n_times)[:,None]*up_factor + np.arange(up_factor)

# %% jupyter={"outputs_hidden": true}
temp

# %%
import scipy
up_temp = scipy.signal.resample(
            x=temp[:,20][None,:],
            num=n_times*up_factor,
            axis=1)
up_temp = up_temp.T

idx = (np.arange(0, n_times)[:,None]*up_factor + np.arange(up_factor))
up_shifted_temps = up_temp[idx].transpose(2,0,1)
up_shifted_temps = np.concatenate(
    (up_shifted_temps,
     np.roll(up_shifted_temps, shift=1, axis=1)),
    axis=2)

# %% jupyter={"outputs_hidden": true}
np.shape(up_shifted_temps)

# %%
ref = np.mean(up_shifted_temps, 0)

# %%
up_shifted_temps = up_shifted_temps.transpose(0,2,1).reshape(-1, n_times)

# %% jupyter={"outputs_hidden": true}
plt.plot(up_shifted_temps.T, c = [0.5, 0.5, 0.5]);
plt.plot(ref, c = 'r')

# %%
nshifts=7
wf_start = nshifts//2
wf_end = -nshifts//2

wf_trunc = ref[None,:][:,wf_start:wf_end]

# %% jupyter={"outputs_hidden": true}
plt.plot(wf_trunc.T);

# %% [markdown]
# Go through the manually picked templates

# %%
manually_picked_temp_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/manual_selected_template_from_benchmark'

# %%
pID

# %%
all_temps = list(Path(manually_picked_temp_dir).glob('*'))
training_temp = []
for i in range(len(Benchmark_pids)):
    pID = Benchmark_pids[i]
    eID, probe = one.pid2eid(pID)
    
    check_dir = main_dir + '/' + 'eID_' + eID + '_probe_' + probe + '_pID_' + pID
    ks_cluster_temp_dir = check_dir + '/ks_cluster_templates.npz'
    ks_cluster_temp = np.load(ks_cluster_temp_dir)
    
    templates_aligned = ks_cluster_temp['templates_aligned']
    
    all_temps = list(Path(manually_picked_temp_dir).glob(pID + '*'))
    
    for j in range(len(all_temps)):
        file_name = all_temps[j].name
        underline_idx = list(re.finditer('_', file_name))
        # pID = file_name[0:underline_idx[0].start()]
        unit_idx = int(file_name[underline_idx[1].end():underline_idx[2].start()])

        temp = templates_aligned[unit_idx - 1]
        training_temp.append(temp)

# %%
training_temp = np.array(training_temp)

# %%
plt.plot(training_temp[50]);

# %%
np.save(manually_picked_temp_dir + '/manual_picked_temp.npy', training_temp)

# %%
list(Path(manually_picked_temp_dir).glob(pID + '*'))

# %%
