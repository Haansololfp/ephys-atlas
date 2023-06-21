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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spike_psvae.waveform_utils import make_channel_index, make_contiguous_channel_index

# %%
spikes_info_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example/1e104bf4-7a24-4624-a5b2-c2c8289c0de7/T00500/spikes.pqt'
ap_dir = '/moto/stats/users/hy2562/projects/ephys_atlas/benchmark_snippets_example/1e104bf4-7a24-4624-a5b2-c2c8289c0de7/T00500/ap.npy'

# %%
df_spikes = pd.read_parquet(spikes_info_dir)
time_points = np.int32(df_spikes['sample'].values)
maxchans = np.int32(df_spikes['trace'].values)


# %%

# %%
extract_box_radius = 200
box_norm_p = 2
extract_channel_index = make_channel_index(
            geom, extract_box_radius, distance_order=False, p=box_norm_p
        )

# %%
