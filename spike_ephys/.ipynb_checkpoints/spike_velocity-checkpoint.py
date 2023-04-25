# %%
import numpy as np
import multiprocessing
import h5py
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.stats import t


# %%
def get_spike_velocity(wf, sp_idx, geom, channel_index, mc, fs):
    wf[np.where(np.isnan(wf))] = 0
    ci = channel_index[mc]

    mcs = np.argmax(wf.ptp(0))
    ci[ci == 384] = ci[mcs]
    local_geom = geom[ci]
    
    local_geom[:, 1] -= local_geom[mcs, 1]
    local_geom[:, 0] -= local_geom[mcs, 0]

    sp_idx = (wf.ptp(0) > 0.12*np.max(wf.ptp(0))) & (wf.ptp(0) > 2) 
    colomn_idx = np.abs((local_geom[:,0] - local_geom[mcs, 0]))<= 16

    above_idx = (local_geom[:,1] >= 0) & colomn_idx
    below_idx = (local_geom[:,1] <= 0) & colomn_idx 
    max_idx = np.nanargmax(np.abs(wf))
    v = wf.ptp(0)
    
    z = local_geom[:,1]
    if np.sign(wf.flatten()[max_idx]) == 1:
        times = np.nanargmax(wf, axis = 0)/fs
        times = times - np.mean(times[z == 0])
    else:
        times = np.nanargmin(wf, axis = 0)/fs
        times = times - np.mean(times[z == 0])
#############        
    z = local_geom[above_idx & sp_idx,1]
    t = times[above_idx & sp_idx]
    
    sorted_z_above = np.sort(z)
    sort_idx = np.argsort(z)
    sorted_t_above = t[sort_idx]
    diff_t_above = np.diff(sorted_t_above)
    
    discard_s_idx = np.where(np.abs(diff_t_above)>0.0005)
    if len(discard_s_idx[0]) == 0:
        keep_idx_above = slice(None)
    else:
        keep_idx_above = np.arange(np.min(discard_s_idx) - 1)  
        
    if (len(np.unique(sorted_z_above[keep_idx_above])) > 2):
        lm1 = LinearRegression(fit_intercept = False)
        lm1.fit(sorted_z_above[keep_idx_above][:,None], sorted_t_above[keep_idx_above])
        velocity_above = lm1.coef_
    else:
        velocity_above = np.nan
        
################
    z = local_geom[below_idx & sp_idx,1]
    t = times[below_idx & sp_idx]

    sorted_z_below = np.sort(z)[::-1]
    sort_idx = np.argsort(z)[::-1]
    sorted_t_below = t[sort_idx]
    diff_t_below = np.diff(sorted_t_below)
    
    discard_s_idx = np.where(np.abs(diff_t_below)>0.0005) #remove outliers that has a peak time of > 0.0005s from the peak time of the previous spot
    if len(discard_s_idx[0]) == 0:
        keep_idx_below = slice(None)
    else:
        keep_idx_below = np.arange(np.min(discard_s_idx) - 1)  

    if (len(np.unique(sorted_z_below[keep_idx_below])) > 2):
        lm2 = LinearRegression(fit_intercept = False)
        lm2.fit(sorted_z_below[keep_idx_below][:,None], sorted_t_below[keep_idx_below])
        velocity_below = lm2.coef_
    else:
        velocity_below = np.nan

    return velocity_above, velocity_below




# %%
def get_spikes_velocity(wfs,
                        geom,
                        maxchans, 
                        spread_idx,
                        channel_index,
                        fs,
                        n_workers=None,
                        # radius = None,
                        # n_channels = None
                       ):
    # maxchans from spike_index in subtraction.h5
    N, T, C = wfs.shape
    maxchans = maxchans.astype(int)
    
    # if n_channels is not None or radius is not None:
    #     subset = channel_index_subset(
    #         geom, channel_index, n_channels=n_channels, radius=radius
    #     )
    # else:
    #     subset = [slice(None)] * len(geom)
        
    xqdm = tqdm

    # -- run the linear regression
    vel_above = np.empty(N)
    vel_below = np.empty(N)
    
    with Parallel(n_workers) as pool:
        
        for n, (v1, v2) in enumerate(
            pool(
                delayed(get_spike_velocity)(
                    wf,
                    sp_idx,
                    geom,
                    channel_index,
                    mc,
                    fs
                )
                for wf, sp_idx, mc in xqdm(
                    zip(wfs, spread_idx, maxchans), total=N, desc="lsq"
                )
            )
        ):

            vel_above[n] = v1
            vel_below[n] = v2

            
    return vel_above, vel_below