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
from pathlib import Path
import os
from glob import glob


# %%
main_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example'

# %%
files = []
start_dir = os.getcwd()
pattern   = "waveforms.npy"

for dir,_,_ in os.walk(main_dir):
    files.extend(glob(os.path.join(dir,pattern))) 

# %%
import re

# %%
slash_idx = [m.start() for m in re.finditer('/', files[0])]

# %%
files[0]

# %%
files[0][slash_idx[-3] + 1:slash_idx[-2]]

# %%
PID = []
T_snippet = []
spk_num = []
for i in range(len(files)):
    wfs = np.load(files[i])
    N, T, C = np.shape(wfs)
    mcs = np.nanargmax(wfs.ptp(1), 1)
    wf = wfs[np.arange(N), :, mcs]
    pos_idx = (np.max(wf, 1) > np.abs(np.min(wf, 1)))
    spk_num.append(np.where(pos_idx)[0])
    
    slash_idx = [m.start() for m in re.finditer('/', files[i])]
    PID.append(files[i][slash_idx[-3] + 1:slash_idx[-2]])
    T_snippet.append(files[i][slash_idx[-2] + 1:slash_idx[-1]])

# %%
d = {'PID': PID, 'T': T_snippet, 'positive_idx': spk_num}

# %%
PID = []

# %%
PID.append(files[i][slash_idx[-3] + 1:slash_idx[-2]])

# %%
PID

# %%
df = pd.DataFrame(data=d)

# %%
positive_spk = np.concatenate(positive_spk)

# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(10, 10, figsize = [20, 20])
for i in range(100):
    r = i//10
    c = np.mod(i, 10)
    axs[r][c].plot(positive_spk[i,:])

# %%
np.save(main_dir + '/positive_spikes.npy', positive_spk)

# %%
import pandas as pd

# %%
df

# %%
df.to_parquet(main_dir + '/positive_spikes.pqt')

# %%
