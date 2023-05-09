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
# For IBL annual meeting

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from brainwidemap import bwm_query
from one.api import ONE

one = ONE()
bwm_df = bwm_query(one)

# %%
import brainwidemap as bwm
from pathlib import Path
bwm.download_aggregate_tables(one, target_path=Path('/moto/stats/users/hy2562/projects/ephys_atlas/bwm'))

# %%
import pandas as pd
df_clusters = pd.read_parquet('/moto/stats/users/hy2562/projects/ephys_atlas/bwm/clusters.pqt')

# %%
df_clusters

# %%
import numpy as np
len(np.unique(df_clusters.uuids))

# %%
