import numpy as np
from scipy.signal import peak_widths
from spike_ephys import spike_velocity
from scipy.spatial.distance import cdist
from scipy.stats import linregress

def peak_value(wfs):
    N, T, C = np.shape(wfs)
    peakvalue = np.zeros((N,))
    for i in range(N):
        a = wfs[i,:,:]
        # a[np.where(np.isnan(a))] = 0
        peakvalue[i]  = max(np.nanmin(a), np.nanmax(a), key=abs)
        if np.isnan(peakvalue[i]):
            print('wrong peak value!')
            raise
    return peakvalue


def ptp_duration(wfs):
    N, T, C = np.shape(wfs)
    ptp_max_idx = np.nanargmax(wfs.ptp(1), axis=1)
    ptp_duration = np.abs(np.nanargmax(wfs, axis = 1) - np.nanargmin(wfs, axis = 1))[np.arange(N), ptp_max_idx]

    return ptp_duration


def halfpeak_duration(wfs):
    N, T, C = np.shape(wfs)        
    peak_max_T = np.nanmax(np.absolute(wfs), axis=1) #N x C
    mcs = np.nanargmax(peak_max_T, axis=1) #N   
    peak_max_id = np.nanargmax(np.abs(wfs[np.arange(N),:,mcs]), axis=1) #N

    half_peak = wfs[np.arange(N), peak_max_id, mcs]/2 #N
    
    peak_sign = np.sign(half_peak) #N
    half_peak = half_peak*peak_sign

    mcs_wfs = wfs[np.arange(N), : , mcs] #N x T
    
    mcs_wfs = np.repeat(peak_sign[:, None], T , axis =1) * mcs_wfs# correct the sign of the waveform

    cross_half_sign = np.sign(mcs_wfs - np.repeat(half_peak[:,None], T, axis=1)) #N x T
    cross_half_diff = np.diff(cross_half_sign) #N x T

    first_cross_diff = np.zeros((N,T - 1))
    second_cross_diff = np.zeros((N,T - 1))
    for i in range(N):
        FCD  = cross_half_diff[i,:].copy() #N x T
        SCD  = cross_half_diff[i,:].copy() #N x T
        
        first_cross_diff[i,0:peak_max_id[i]] = np.squeeze(np.flip(FCD[None,0:peak_max_id[i]], axis = 1))
        second_cross_diff[i,0:(T-1-peak_max_id[i])] =  SCD[peak_max_id[i]:None]
    
    last_cross =(second_cross_diff == -2).argmax(axis=1)
    
    first_cross =(first_cross_diff == 2).argmax(axis=1)

    return last_cross + first_cross


def peak_trough_ratio(wfs):
    mcs = np.nanargmax(wfs.ptp(1), axis=1)
    maxchan_traces = wfs[np.arange(np.shape(wfs)[0]), :, mcs]
    trough_depths = maxchan_traces.min(1)
    peak_heights = maxchan_traces.max(1)

    PT_ratio = np.log(-peak_heights/trough_depths)

    return PT_ratio



def spatial_spread(wfs, geom, ci, spk_idx, threshold = 0.25):
    # threshold by ptps, average of distance to maxchan
    N, T, C = np.shape(wfs)
        
    wfs_ptp = wfs.ptp(1)
    max_ptp = np.nanmax(wfs_ptp, axis = 1)
    mcs = np.nanargmax(wfs_ptp, axis = 1)
    
    threshold_ptp = max_ptp * threshold
    
    dist = np.zeros((N, C))
    for i in range(N):
        local_ci = ci[spk_idx[i]]
        local_ci[local_ci == 384] = local_ci[mcs[i]]
        dist_i = cdist(geom[local_ci], geom[local_ci[mcs[i]]][None,:])
        dist[i,:] = np.squeeze(dist_i)
    
    spatial_spread = np.nanmean(dist*(1 + np.sign(wfs_ptp - np.repeat(threshold_ptp[:, None], C, axis = 1)))/2, axis = 1)

    return spatial_spread
 


def reploarization_slope(wfs, fs, window = 50):
    #win = 2*resample/fs
    N, T, C = np.shape(wfs)
    time = np.arange(T)/fs * 1000
    wfs_ptp = wfs.ptp(1)
    mcs = np.nanargmax(wfs_ptp, axis = 1)
    
    wfs = wfs[np.arange(N),:,mcs]
    
    peakvalue  = np.nanmax(wfs, axis = 1)
    troughvalue = np.nanmin(wfs, axis = 1)
    
    peak_idx = np.nanargmax(wfs, axis = 1)
    trough_idx = np.nanargmin(wfs, axis = 1)

    reverse_vec = np.abs(troughvalue)> np.abs(peakvalue) # deal with positive spikes

    rep_slope = np.zeros(N)
    for i in range(N):
        if reverse_vec[i]:
            rep_slope[i] = linregress(time[trough_idx[i]:trough_idx[i]+window],wfs[i, trough_idx[i]:trough_idx[i]+window])[0]
        else:
            rep_slope[i] = -linregress(time[peak_idx[i]:peak_idx[i]+window],wfs[i, peak_idx[i]:peak_idx[i]+window])[0]
    return rep_slope


def recovery_slope(wfs, fs, window = 50):   
    N, T, C = np.shape(wfs)
    time = np.arange(T)/fs * 1000 #transfer to time in ms
    wfs_ptp = wfs.ptp(1)
    mcs = np.nanargmax(wfs_ptp, axis = 1)
    
    wfs = wfs[np.arange(N),:,mcs]
    
    peakvalue  = np.nanmax(wfs, axis = 1)
    troughvalue = np.nanmin(wfs, axis = 1)
    
    peak_idx = np.nanargmax(wfs, axis = 1)
    trough_idx = np.nanargmin(wfs, axis = 1)

    reverse_vec = np.abs(troughvalue)> np.abs(peakvalue)
        
    summit_idx = np.concatenate((peak_idx[:,None], trough_idx[:,None]), axis = 1)

    summit_idx = summit_idx[np.arange(N), reverse_vec.astype(int)]
    
    rec_slope = np.zeros(N)
    for i in range(N):
        
        if reverse_vec[i]:
            s_idx = np.nanargmax(wfs[i, summit_idx[i]:None])
            s_idx = s_idx + summit_idx[i]
            rec_slope[i] = linregress(time[s_idx:s_idx+window],wfs[i, s_idx:s_idx+window])[0]
        else:
            s_idx = np.nanargmin(wfs[i, summit_idx[i]:None])
            s_idx = s_idx + summit_idx[i]
            rec_slope[i] = -linregress(time[s_idx:s_idx+window],wfs[i, s_idx:s_idx+window])[0]
            
    return rec_slope


def spatial_spread_weighted_dist(wfs, geom, ci, spk_idx):
    N, T, C = np.shape(wfs)
    wfs_ptp = wfs.ptp(1)
    mcs = np.nanargmax(wfs_ptp, axis = 1)
    
    spatial_spread = np.zeros(N)
    for i in range(N):
        local_ci = ci[spk_idx[i]]
        local_ci[local_ci == 384] = local_ci[mcs[i]]
        dist_i = np.squeeze(cdist(geom[local_ci], geom[local_ci[mcs[i]]][None,:]))
        spatial_spread[i] = np.nansum(dist_i[None,:]*wfs_ptp[i,:])/np.nansum(wfs_ptp[i,:])
    
    return spatial_spread



def velocity(wfs, geom, channel_index, fs, mcs, n_workers = 4, threshold = 0.12):
    N, T, C = np.shape(wfs)
    wfs_ptp = wfs.ptp(1)
    max_ptp = np.nanmax(wfs_ptp, axis = 1)

    # mcs = np.nanargmax(wfs_ptp, axis=1)

    threshold_ptp = max_ptp * threshold

    spread_idx = (wfs_ptp - np.repeat(threshold_ptp[:, None], C, axis = 1)) > 0

    vel_above, vel_below = spike_velocity.get_spikes_velocity(
        wfs,
        geom,
        mcs, 
        spread_idx,
        channel_index,
        fs, 
        n_workers=n_workers,
    )

    return vel_above, vel_below



def depolarization_slope(wfs, fs, window = 50):
    #win = 2*resample/fs
    N, T, C = np.shape(wfs)
    time = np.arange(T)/fs * 1000
    wfs_ptp = wfs.ptp(1)
    mcs = np.nanargmax(wfs_ptp, axis = 1)
    
    wfs = wfs[np.arange(N),:,mcs]
    
    peakvalue  = np.nanmax(wfs, axis = 1)
    troughvalue = np.nanmin(wfs, axis = 1)
    
    peak_idx = np.nanargmax(wfs, axis = 1)
    trough_idx = np.nanargmin(wfs, axis = 1)

    reverse_vec = np.abs(troughvalue)> np.abs(peakvalue) # deal with positive spikes

    dep_slope = np.zeros(N)
    for i in range(N):
        if reverse_vec[i]:
            dep_slope[i] = linregress(time[trough_idx[i]-window:trough_idx[i]],wfs[i, trough_idx[i]-window:trough_idx[i]])[0]
        else:
            dep_slope[i] = -linregress(time[peak_idx[i]-window:peak_idx[i]],wfs[i, peak_idx[i]-window:peak_idx[i]])[0]
    return dep_slope