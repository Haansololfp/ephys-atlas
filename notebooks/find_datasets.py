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
# find closest trajectory

# %%
# %load_ext autoreload
# %autoreload 2

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
i = 1
pid = Benchmark_pids[i]
one = ONE(base_url='https://alyx.internationalbrainlab.org')
eID, probe = one.pid2eid(pid)

# %%
sess = one.alyx.rest('sessions', 'list', 'probe_insertion' == pid)

# %%
sess[0]

# %%
# Author: Mayo Faulkner
# import modules
import numpy as np
from one.api import ONE

import ibllib.pipes.histology as histology
import ibllib.atlas as atlas
import pandas as pd


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ibllib.atlas import BrainRegions
from brainbox.ephys_plots import plot_brain_regions
# from atlaselectrophysiology import rendering

# %%
one.alyx.rest('sessions','list', subject='PL035', date='2023-04-11', probe='probe00')

# %%
one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                               subject=subject, date=date, probe=probe_label)

# %% jupyter={"outputs_hidden": true}
# Instantiate brain atlas and one
brain_atlas = atlas.AllenAtlas(25)

# Find all trajectories with histology tracing
all_hist = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track')
# Some do not have tracing, exclude these ones
sess_with_hist = [sess for sess in all_hist if sess['x'] is not None]
traj_ids = [sess['id'] for sess in sess_with_hist]
# Compute trajectory objects for each of the trajectories
trajectories = [atlas.Insertion.from_dict(sess) for sess in sess_with_hist]

# Find the trajectory of the id that you want to find closeby probe insertions for
subject = sess[0]['subject']
date = sess[0]['start_time'][0:10]
probe_label = probe
traj_origin_id = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                               subject=subject, date=date, probe=probe_label)[0]['id']
# Find the index of this trajectory in the list of all trajectories
chosen_traj = traj_ids.index(traj_origin_id)

# Define active part of probe ~ 200um from tip and ~ (200 + 3900)um to top of channels
depths = np.arange(200, 4100, 20) / 1e6
traj_coords = np.empty((len(traj_ids), len(depths), 3))

# For each trajectory compute the xyz coords at positions depths along trajectory
for iT, traj in enumerate(trajectories):
    traj_coords[iT, :] = histology.interpolate_along_track(np.vstack([traj.tip, traj.entry]),
                                                           depths)

# Find the average distance between all positions compared to trjaectory of interest
avg_dist = np.mean(np.sqrt(np.sum((traj_coords - traj_coords[chosen_traj]) ** 2, axis=2)), axis=1)

# Sort according to those that are closest
closest_traj = np.argsort(avg_dist)

close_sessions = dict()
# Make a 3D plot showing trajectory of interest (in black) and the 10 nearest trajectories (blue)
# fig = rendering.figure(grid=False)

for iSess, sess_idx in enumerate(closest_traj[0:10]):

#     mlapdv = brain_atlas.xyz2ccf(traj_coords[sess_idx])
#     if iSess == 0:
#         mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
#                     line_width=1, tube_radius=10, color=(0, 0, 0))
#     else:
#         mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
#                     line_width=1, tube_radius=10, color=(0.0, 0.4, 0.5))

#     mlab.text3d(mlapdv[0, 1], mlapdv[0, 2], mlapdv[0, 0], str(iSess),
#                 line_width=4, color=(0, 0, 0), figure=fig, scale=150)

    close_sessions[iSess] = dict()
    close_sessions[iSess]['subject'] = sess_with_hist[sess_idx]['session']['subject']
    close_sessions[iSess]['date'] = sess_with_hist[sess_idx]['session']['start_time'][:10]
    close_sessions[iSess]['probe'] = sess_with_hist[sess_idx]['probe_name']
    close_sessions[iSess]['dist'] = avg_dist[closest_traj[iSess]] * 1e6

    # close_sessions.append((sess_with_hist[sess_idx]['session']['subject'] + ' ' +
    #                        sess_with_hist[sess_idx]['session']['start_time'][:10] +
    #                        ' ' + sess_with_hist[sess_idx]['probe_name'] + ': dist = ' +
    #                        str(avg_dist[closest_traj[iSess]] * 1e6)))

# print(close_sessions)

# %%
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")
df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))

# %%
np.shape(np.where(df_clusters['ks2_label'] == 'good'))

# %%
np.shape(np.where((df_clusters['label'] >0.5) & (df_clusters['acronym'] == 'MRN')))

# %%

# %%
np.sum(df_clusters['spike_count'].values)

# %%
df_clusters = df_clusters.reset_index(drop=False)
len(np.unique(df_clusters.pid.values))

# %%
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
df_channels.pid[df_channels['acronym'] == 'ACAd5']

# %%
np.unique(df_channels.pid[df_channels['acronym'] == 'ACAd5'].values)

# %%
len(np.unique(df_channels['acronym'].values))

# %%
len(np.unique(df_channels.pid.values))
# np.unique(df_channels.pid.values)

# %%
regions = np.unique(df_channels['acronym'].values)
n_pid_array = []
n_neuron_array = []
for i in range(len(regions)):
    region = regions[i]
    n_pid = np.shape(np.unique(df_channels.pid[df_channels['acronym'] == region].values))[0]
    n_neuron = np.shape(np.where((df_clusters['label'] >0.5) & (df_clusters['acronym'] == region)))[1]
    n_pid_array.append(n_pid)
    n_neuron_array.append(n_neuron)

# %%
fig, axes = plt.subplots(1, 3, figsize = (30, 10))
axes[0].hist(n_neuron_array, bins = np.arange(0, 2010, 10));
axes[1].hist(n_pid_array, bins = np.arange(200));
axes[2].hist(np.divide(n_neuron_array,n_pid_array), bins = np.arange(60));


# %%
average_neurons = np.divide(n_neuron_array,n_pid_array)
n_pid_array = np.array(n_pid_array)

regions[np.where(average_neurons>20)][n_pid_array[np.where(average_neurons>20)]>10]

# %% jupyter={"outputs_hidden": true}
output_folder = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_represented_regions_allen_visualize'

highly_represented_regions = regions[np.where(average_neurons>20)][n_pid_array[np.where(average_neurons>20)]>10]
br = BrainRegions()
for i in range(len(highly_represented_regions)):
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    df_channels = df_channels.reset_index(drop=False)
    
    RL = highly_represented_regions[i]
    pids = np.unique(df_channels.pid[df_channels['acronym'] == RL].values)
    
    l = len(pids)
    
    rows = np.int32(np.ceil(l/10))
    
    fig, axs = plt.subplots(rows, 10, figsize = (70, rows*15))
    fig.suptitle(RL, fontsize = 50)
    for j in range(l):
        pID = pids[j]
        df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

        df_channels = df_channels.reset_index(drop=False)
        df_channels = df_channels[df_channels.pid == pID]
        df_channels = df_channels.reset_index(drop=True)

        row = j//10
        col = np.mod(j, 10)
        
        if rows>1:
            plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[row][col], title=pID, label='right')
        else:
            plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[col], title=pID, label='right') 
    if '/' in RL:
        RL.replace('/','_')
    plt.savefig(output_folder + '/brain_regions_plot_' + RL + '.png')
    plt.close()

# %% jupyter={"outputs_hidden": true}
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
pids = np.unique(df_channels.pid[df_channels['acronym'] == 'CA1'].values)
fig, axs = plt.subplots(7, 10, figsize = (60, 100))
for i in range(len(pids)):
    pID = pids[i]
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    
    br = BrainRegions()

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)
    
    row = i//10
    col = np.mod(i, 10)
    # try:
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                       brain_regions=br, display=True, ax=axs[row][col], title='Real', label='right')
    # except:
    #     continue

# %% jupyter={"outputs_hidden": true}
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
pids = np.unique(df_channels.pid[df_channels['acronym'] == 'CP'].values)
fig, axs = plt.subplots(7, 10, figsize = (60, 100))
for i in range(len(pids)):
    pID = pids[i]
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    
    br = BrainRegions()

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)
    
    row = i//10
    col = np.mod(i, 10)
    # try:
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                       brain_regions=br, display=True, ax=axs[row][col], title='Real', label='right')
    # except:
    #     continue

# %% jupyter={"outputs_hidden": true}
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
pids = np.unique(df_channels.pid[df_channels['acronym'] == 'ACAd5'].values)
fig, axs = plt.subplots(7, 10, figsize = (60, 100))
for i in range(len(pids)):
    pID = pids[i]
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    
    br = BrainRegions()

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)
    
    row = i//10
    col = np.mod(i, 10)
    # try:
    plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                       brain_regions=br, display=True, ax=axs[row][col], title='Real', label='right')
    # except:
    #     continue

# %% jupyter={"outputs_hidden": true}
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ibllib.atlas import BrainRegions
from brainbox.ephys_plots import plot_brain_regions

LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")

one = ONE(base_url="https://alyx.internationalbrainlab.org")
fig, axs = plt.subplots(1, len(close_sessions), figsize = (60, 20))

for i in range(len(close_sessions)):
    subject = close_sessions[i]['subject']
    date = close_sessions[i]['date']
    probe_label = close_sessions[iSess]['probe']
    eID = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                               subject=subject, date=date, probe=probe_label)[0]['session']['id']
    pID_array = one.eid2pid(eID)
    
    a = pID_array[1]
    indexes = [index for index in range(len(a)) if a[index] == probe]
    
    pID = pID_array[0][indexes[0]]
    
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    

    br = BrainRegions()


    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)
    
    # plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
    #                        brain_regions=br, display=True, ax=axs[i], title='Real', label='right')
    
    try:
        plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[i], title='Real', label='right')
    except:
        continue

# %%
eID = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                               subject='NYU-27', date='2020-09-30', probe='probe00')[0]['session']['id']
pID_array = one.eid2pid(eID)

# %%
import csv

aids_array = []
n_neurons = []
n_pid = []
acronyms_array = []
i = 0
df_save_ins_region_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/df_save_goodunit_ins_region.csv'
with open(df_save_ins_region_dir, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if i>0:
            a = ', '.join(row)
            idx = [x.start() for x in re.finditer(',', a)]
            aids_array.append(int(a[0:idx[0]]))
            n_neurons.append(int(a[idx[0]+1:idx[1]]))
            n_pid.append(int(a[idx[1]+1:idx[2]]))
            acronyms_array.append(a[idx[2]+1:None])
        i += 1

# %% jupyter={"outputs_hidden": true}
fig, axes = plt.subplots(1, 3, figsize = (30, 10))
axes[0].hist(n_neurons, bins = np.arange(0, 2010, 10));
axes[1].hist(n_pid, bins = np.arange(200));
axes[2].hist(np.divide(n_neurons,n_pid), bins = np.arange(60));

# %%
n_pid = np.array(n_pid)
acronyms_array = np.array(acronyms_array)

average_neurons = np.divide(n_neurons,n_pid)
dense_idx = np.where(average_neurons>20)
frequent_idx = np.where(n_pid>10)

picked_idx = np.intersect(dense_idx, frequent_idx)

for i in range(len(picked_idx)):
    idx = picked_idx[i]
    acronym = acronyms_array[idx]
    

# %%
average_neurons = np.divide(n_neurons,n_pid)


# %%
acronyms_array[np.where(average_neurons>20)][n_pid[np.where(average_neurons>20)]>10]


# %% [markdown]
# find highly represented beryl

# %%
br = BrainRegions()

# %%
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")
df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
Beryl_regions = br.remap(df_channels['atlas_id'], source_map='Allen', target_map='Beryl')

# %%
len(np.unique(br.id2acronym(Beryl_regions)))

# %%
br.id2acronym(np.unique(Beryl_regions))

# %%
df_channels = df_channels.reset_index(drop=False)

# %%
regions = np.unique(Beryl_regions)
n_pid_array = []
n_neuron_array = []
n_ks_label_array = []
cluster_Beryl_regions = br.remap(df_clusters['atlas_id'], source_map='Allen', target_map='Beryl')

for i in range(len(regions)):
    region = regions[i]
    n_pid = np.shape(np.unique(df_channels.pid[Beryl_regions == region].values))[0]
    n_neuron_perfect = np.shape(np.where((df_clusters['label'] ==1) & (cluster_Beryl_regions == region)))[1]
    n_neuron_good = np.shape(np.where((df_clusters['label'] >0.5) & (cluster_Beryl_regions == region)))[1]
    n_neuron_meh = np.shape(np.where((df_clusters['label'] >0.3) & (cluster_Beryl_regions == region)))[1]
    
    n_ks_label = np.shape(np.where((df_clusters['ks2_label'] == 'good')& (cluster_Beryl_regions == region)))[1]
    
    n_pid_array.append(n_pid)
    n_neuron_array.append([n_neuron_perfect, n_neuron_good - n_neuron_perfect, n_neuron_meh - n_neuron_good])
    n_ks_label_array.append(n_ks_label)

# %%
fig, axes = plt.subplots(1, 3, figsize = (30, 10))
axes[0].hist(n_neuron_array, bins = np.arange(0, 2010, 10));
axes[1].hist(n_pid_array, bins = np.arange(200));
axes[2].hist(np.divide(n_neuron_array,n_pid_array), bins = np.arange(60));


# %%
average_neurons = np.divide(n_neuron_array,n_pid_array)
n_pid_array = np.array(n_pid_array)

br.id2acronym(regions[np.where(average_neurons>20)][n_pid_array[np.where(average_neurons>20)]>20])

# %%
output_folder = '/moto/stats/users/hy2562/projects/ephys_atlas/highly_represented_regions_beryl_visualize'

highly_represented_regions = regions[np.where(average_neurons>20)][n_pid_array[np.where(average_neurons>20)]>10]
br = BrainRegions()
for i in range(101,len(highly_represented_regions)):
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    df_channels = df_channels.reset_index(drop=False)
    
    RL = highly_represented_regions[i]
    pids = np.unique(df_channels.pid[Beryl_regions == RL].values)
    
    l = np.min([len(pids), 100])
    
    rows = np.int32(np.ceil(l/10))
    
    fig, axs = plt.subplots(rows, 10, figsize = (70, rows*15))
    fig.suptitle(RL, fontsize = 50)
    for j in range(l):
        pID = pids[j]
        df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

        df_channels = df_channels.reset_index(drop=False)
        df_channels = df_channels[df_channels.pid == pID]
        df_channels = df_channels.reset_index(drop=True)

        row = j//10
        col = np.mod(j, 10)
        
        if rows>1:
            plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[row][col], title=pID, label='right')
        else:
            plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[col], title=pID, label='right')         
    plt.savefig(output_folder + '/brain_regions_plot_' + br.id2acronym(RL)[0] + '.png')
    plt.close()

# %%
br.id2acronym(np.unique(Beryl_regions))

# %%
import matplotlib.pyplot as plt
import numpy as np
which = np.arange(1, len(np.unique(br.id2acronym(Beryl_regions))))
# Data for the first subplot
labels1 = br.id2acronym(np.unique(Beryl_regions))[which]
category1 = ['perfect', 'good', 'meh']
values1 = np.array(n_neuron_array)[which,:]
sort_idx = np.argsort(values1[:,0])
labels1 = labels1[sort_idx]
values1 = values1[sort_idx,:]
# Data for the second subplot
labels2 = br.id2acronym(np.unique(Beryl_regions))[which]
values2 = np.array(n_pid_array)[which]
sort_idx2 = np.argsort(values2)
labels2 = labels2[sort_idx2]
values2 = values2[sort_idx2]


# Data for the third subplot
labels3 = br.id2acronym(np.unique(Beryl_regions))[which]
values3 = np.array(n_ks_label_array)[which]
sort_idx3 = np.argsort(values3)
labels3 = labels3[sort_idx3]
values3 = values3[sort_idx3]
# Set up the figure and axes
fig, axs = plt.subplots(1, 3, figsize=(10, 8))

# Create the first subplot
for i in range(len(category1) - 1):
    axs[0].barh(labels1, values1[:,i], height=0.8, left=np.sum(values1[:,:i], axis=1),
               color=['#DAF7A6', '#FFC300','#FF5733'][i], label=category1[i])
    # for j, v in enumerate(values1[:,i]):
    #     axs[0].text(np.sum(values1[:,:i], axis=1)[j] + v/2, j, str(v), color='black', fontweight='bold')
axs[0].set_title('# of perfect units:' + str(np.sum(values1[:,0])) + ', good units:' + str(np.sum(values1[:,1])))
axs[0].set_xlabel('#')
axs[0].set_ylabel('Beryl Region')
axs[0].legend()


def tick_formatter1(x, pos):
    if (len(labels1) - 1 - pos) % 10 == 0:
        return labels1[pos]
    else:
        return ''
    
def tick_formatter2(x, pos):
    if (len(labels2) - 1 - pos) % 10 == 0:
        return labels2[pos]
    else:
        return ''
    
def tick_formatter3(x, pos):
    if (len(labels3) - 1 - pos) % 10 == 0:
        return labels3[pos]
    else:
        return ''

# Set the y-axis tick formatter
axs[0].yaxis.set_major_formatter(plt.FuncFormatter(tick_formatter1))


# Create the second subplot
axs[1].barh(labels3, values3, height=0.8,
           color=['#FF5733'])
    # for j, v in enumerate(values2[:,i]):
    #     axs[1].text(np.sum(values2[:,:i], axis=1)[j] + v/2, j, str(v), color='black', fontweight='bold')
axs[1].set_title('# of good ks units: ' + str(np.sum(values3)))
axs[1].set_xlabel('#')
axs[1].set_ylabel('Beryl Region')
axs[1].yaxis.set_major_formatter(plt.FuncFormatter(tick_formatter3))


###

# Create the second subplot
axs[2].barh(labels2, values2, height=0.8,
           color=['#FF5733'])
    # for j, v in enumerate(values2[:,i]):
    #     axs[1].text(np.sum(values2[:,:i], axis=1)[j] + v/2, j, str(v), color='black', fontweight='bold')
axs[2].set_title('# of pIDs: ' + str(len(np.unique(df_channels.pid.values))))
axs[2].set_xlabel('#')
axs[2].set_ylabel('Beryl Region')
axs[2].yaxis.set_major_formatter(plt.FuncFormatter(tick_formatter2))
# Adjust the layout and display the plot
plt.tight_layout()
plt.savefig('dataset_summation_stats_20230414')
plt.show()


# %% [markdown]
# find the set of recordings that covers the regions that we are interested in and also share the most regions

# %%
Beryl_regions_to_look_at = ['CA1' , 'ANcr2', 'LGd', 'APN', 'LP']
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
Beryl_regions = br.id2acronym(br.remap(df_channels['atlas_id'], source_map='Allen', target_map='Beryl'))

# %%
df_channels = df_channels.reset_index(drop=False)
pids_all = df_channels.pid.values
Beryl_regions_all = np.unique(br.remap(df_channels['atlas_id'], source_map='Allen', target_map='Beryl'))
Beryl_regions = br.id2acronym(br.remap(df_channels['atlas_id'], source_map='Allen', target_map='Beryl'))
for i in range(len(Beryl_regions_to_look_at)):
    RL = Beryl_regions_to_look_at[i]
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    df_channels = df_channels.reset_index(drop=False)
    
    pids = np.unique(df_channels.pid[Beryl_regions == RL].values)
    
    coverage_matrix = np.zeros((len(pids), len(Beryl_regions_all)))
    all_idx = []
    for j in range(len(pids)):
        pid = pids[j]
        entries_of_interest = np.where([df_channels.pid.values == pid])[1]
        df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
        df_channels = df_channels.reset_index(drop=False)
        allen_coverage = df_channels.atlas_id[entries_of_interest]
        beryl_coverage = np.unique(br.remap(allen_coverage, source_map='Allen', target_map='Beryl'))
        for k in range(len(beryl_coverage)):
            RL = beryl_coverage[k]
            idx = np.where(Beryl_regions_all == RL)
            all_idx.append(idx)
            coverage_matrix[j, idx] = 1
    corr = np.corrcoef(coverage_matrix[:,np.unique(all_idx)])
    
    coh_vec = np.sum(corr > 0.8, axis = 0)
    n_coh = np.max(coh_vec)
    coh_pids_idx = np.squeeze(np.where(coh_vec >= n_coh - 10))
    
    plt.figure()
    plt.imshow(np.corrcoef(coverage_matrix[:,np.unique(all_idx)]))
    plt.colorbar()
    
    plt.figure()
    l = len(coh_pids_idx)
    
    rows = np.int32(np.ceil(l/10))
    
    fig, axs = plt.subplots(rows, 10, figsize = (70, rows*15))
    fig.suptitle(RL, fontsize = 50)
    
    for j in range(l):
        pID = pids[coh_pids_idx[j]]
        df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

        df_channels = df_channels.reset_index(drop=False)
        df_channels = df_channels[df_channels.pid == pID]
        df_channels = df_channels.reset_index(drop=True)

        row = j//10
        col = np.mod(j, 10)
        
        if rows>1:
            plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[row][col], title=pID, label='right')
        else:
            plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=axs[col], title=pID, label='right') 

        
        

# %%
pids_cohort = pids[coh_pids_idx]
    

# %%
plt.imshow(coverage_matrix[:,np.unique(all_idx)][coh_pids_idx,:])

# %%
region_coverage = coverage_matrix[:,np.unique(all_idx)][coh_pids_idx,:]
region_coverage_frequency = np.sum(coverage_matrix[:,np.unique(all_idx)][coh_pids_idx,:], axis = 0)
remove_idx = np.where((region_coverage_frequency<5) & (region_coverage_frequency>0))

# %%
remove_idx = np.squeeze(remove_idx)

# %%
remove_pid_idx = []
for i in range(len(coh_pids_idx)):
    for j in range(len(remove_idx)):
        if region_coverage[i, remove_idx[j]]:
            remove_pid_idx.append(i)
remove_pid_idx  = np.unique(np.array(remove_pid_idx))

# %%
keep_idx = np.delete(coh_pids_idx, remove_pid_idx)

# %%
plt.imshow(coverage_matrix[:,np.unique(all_idx)][keep_idx,:])

# %%
plt.figure()
l = len(keep_idx)

rows = np.int32(np.ceil(l/10))

fig, axs = plt.subplots(rows, 10, figsize = (70, rows*15))
fig.suptitle(RL, fontsize = 50)

for j in range(l):
    pID = pids[keep_idx[j]]
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))

    df_channels = df_channels.reset_index(drop=False)
    df_channels = df_channels[df_channels.pid == pID]
    df_channels = df_channels.reset_index(drop=True)

    row = j//10
    col = np.mod(j, 10)

    if rows>1:
        plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                       brain_regions=br, display=True, ax=axs[row][col], title=pID, label='right')
    else:
        plot_brain_regions(br.remap(df_channels['atlas_id'].values, source_map='Allen', target_map='Beryl'), channel_depths=df_channels['axial_um'].values,
                       brain_regions=br, display=True, ax=axs[col], title=pID, label='right') 


        

# %%
highly_coverage_pids = pids[keep_idx]
np.save('small_set_pids.npy', highly_coverage_pids)

# %%
from collections import Counter

# %%
Counter(df_clusters['acronym'][df_clusters['channels']])

# %%
# %load_ext autoreload
# %autoreload 2
from spike_ephys import meta_bwm
meta_bwm.plot_bar_neuron_count()
# plt.savefig('ephys_atlas_meta_summary_NpID.png', dpi=2000)

# %%
LOCAL_DATA_PATH = Path("/moto/stats/users/hy2562/projects/ephys_atlas")
df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))

# %%
df_clusters

# %%
df_clusters['channels'].values

# %%
import csv

# %%
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_channels = df_channels.reset_index(drop=False)
pids_all = df_channels.pid.values
Beryl_regions_all = np.unique(br.remap(df_channels['atlas_id'], source_map='Allen', target_map='Beryl'))
Beryl_regions = br.id2acronym(br.remap(df_channels['atlas_id'], source_map='Allen', target_map='Beryl'))

Beryl_regions_covered = br.id2acronym(Beryl_regions_all, mapping = 'Beryl')
with open('table.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Beryl', 'Cosmos','# recordings', '# neurons', '# good neurons'])
    writer.writeheader()
for i in range(len(Beryl_regions_covered)):
    Beryl_RL = Beryl_regions_covered[i]
    if (Beryl_RL == 'root') | (Beryl_RL == 'void'):
        continue
    region = Beryl_regions_all[i]
    Cosmos_RL = br.id2acronym(br.remap(region, source_map='Beryl', target_map='Cosmos'))[0]
    n_pid = np.shape(np.unique(df_channels.pid[Beryl_regions == Beryl_RL].values))[0]
    
    n_neuron_good = np.shape(np.where((df_clusters['label'] ==1) & (cluster_Beryl_regions == region)))[1]
    n_neuron = np.shape(np.where(cluster_Beryl_regions == region))[1]
    with open('table.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([Beryl_RL, Cosmos_RL, n_pid, n_neuron, n_neuron_good])


# %%
meta_bwm.neuron_number_swansons()
plt.savefig('ephys_atlas_swansons_map.png')

# %%
