import numpy as np
import torch
from multiprocess import Process, Manager

def roll_by_gather(mat, shifts: torch.LongTensor):
    '''
    roll the torch waveforms by different amount on different channels
    '''
    N,T = mat.shape 
    arange1 = torch.arange(T).view((1,T)).repeat((T,1))
    arange2 = (arange1 - shifts) % n_cols
    return torch.gather(mat, 1, arange2)

def wfs_corr(wfs_raw, wfs_denoise):
    return torch.sum(wfs_denoise*wfs_raw, 1)/torch.sqrt(torch.sum(wfs_raw*wfs_raw,1) * torch.sum(wfs_denoise*wfs_denoise,1))

def ptp(t, axis):
    # ptp for torch
    t = torch.nan_to_num(t, nan=0.0)
    return t.max(axis).values - t.min(axis).values

def denoise_with_phase_shift(chan_wfs, phase_shift, dn, spk_signs, offset=42, small_threshold = 2, corr_th = 0.8):
    '''
    input an NxT matrix of spike wavforms, and return 1) the denoised waveforma according to the phaseshift 2) the phaseshift of the denoised waveform, 3) index that shows whether the denoised waveform is identified as hallucination
    '''
    N, T = chan_wfs.shape
    
    chan_wfs = roll_by_gather(chan_wfs, - int(phase_shift))
    
    wfs_denoised = dn(chan_wfs)
    
    which = slice(offset-10, offset+10)
    
    d_s_corr = wfs_corr(chan_wfs[:, which], wfs_denoised[:, which])
    halu_idx = (ptp(wfs_denoised, 1)<small_threshold) & (d_s_corr<corr_th)
    
    wfs_denoised = torch.roll(wfs_denoised, phase_shift)
    
    #CHECK THE CORRELATION BETWEEN THE DENOISED WAVEFORM AND THE RAW WAVEFORM, HALLUCINATION WILL HAVE A SMALL VALUE
    phase_shifted = torch.argmax(torch.swapaxes(wfs_denoised, 0, 1) * spk_signs, 0) - offset
    phase_shifted[halu_idx] = 0

    return wfs_denoised, phase_shifted.long(), halu_idx.long()


def make_ci_graph(channel_index, geom, device, CH_N=384):

    channel_index = torch.tensor(channel_index).to(device)
    geom = torch.tensor(geom).to(device)
    CH_N = torch.tensor(CH_N).to(device)
    
    N, L = channel_index.shape
    x_pitch = torch.diff(torch.unique(geom[:, 0]))[0]
    y_pitch = torch.diff(torch.unique(geom[:, 1]))[0]

    ci_graph_all = {}
    maxCH_neighbor = torch.ones((CH_N, 8)) * (L - 1)  # used a hack here, to make sure the maxchan neighbor wfs is zero if index out of the probe
    for i in range(N):
        ci = channel_index[i]
        non_nan_idx = torch.where(ci < CH_N)[0]
        ci = ci[non_nan_idx]
        l = len(ci)
        ci_graph = {}
        ci_geom = geom[ci]
        for ch in range(l):
            group = torch.where(((torch.abs(ci_geom[:, 0] - ci_geom[ch, 0]) == x_pitch) & (torch.abs(ci_geom[:, 1] - ci_geom[ch, 1]) == y_pitch)) |
                                ((torch.abs(ci_geom[:, 0] - ci_geom[ch, 0]) == 0) & (torch.abs(ci_geom[:, 1] - ci_geom[ch, 1]) == 2 * y_pitch)) |
                                ((torch.abs(ci_geom[:, 0] - ci_geom[ch, 0]) == 2 * x_pitch) & (torch.abs(ci_geom[:, 1] - ci_geom[ch, 1]) == 0)))[0]
            ci_graph[ch] = group

        maxCH_idx = torch.where(ci == i)[0]
        maxCH_n = ci_graph[maxCH_idx[0].item()]
        maxCH_neighbor[i, 0:(len(maxCH_n) + 1)] = torch.cat([maxCH_n, maxCH_idx])

        ci_graph_all[i] = ci_graph

    return ci_graph_all, maxCH_neighbor

def make_spike_specific_ci_graph(ci_graph_all, i, ci_graph_on_probe, maxchans, real_maxCH):
    ci_graph =  dict()
    l = len(ci_graph_on_probe[maxchans[i]])
    mcs_idx = real_maxCH[i]
    
    for ch in range(l):
        group = ci_graph_on_probe[maxchans[i]][ch].clone().detach()

        if (len(torch.nonzero(group > mcs_idx))!=0) & (len(torch.nonzero(group < mcs_idx))!=0) & (ch!= mcs_idx):
            if ch>mcs_idx:
                ci_graph[ch] = torch.cat([group[group > mcs_idx], torch.tensor([mcs_idx], device = device)])
            else:
                ci_graph[ch] = torch.cat([group[group < mcs_idx], torch.tensor([mcs_idx], device = device)])
        else:
            ci_graph[ch] = group
            
    ci_graph_all[i] = ci_graph

def multichan_phase_shift_denoise(waveforms, ci_graph_on_probe, maxCH_neighbor, Denoiser, maxchans, CH_N=384, offset=42):
    t = time.time()
    N, T, C = waveforms.shape
    
    if waveforms.get_device() >= 0:
        device = "cuda"
    else:
        device = "cpu"
    
    DenoisedWF = torch.zeros(waveforms.shape, device = device)
        

    col_idx = maxCH_neighbor[maxchans, :]
    row_idx = torch.arange(N)[None,:].repeat(8, 1)
    maxCH_neighbor_wfs = waveforms[torch.reshape(row_idx.T,(-1,)), : , torch.flatten(col_idx)]
    wfs_denoised_mc_neighbors = Denoiser(maxCH_neighbor_wfs).reshape([N, 8, T])
    # wfs_denoised_mc_neighbors = torch.nan_to_num(wfs_denoised_mc_neighbors, nan=0.0)
    max_neighbor_ptps = ptp(wfs_denoised_mc_neighbors, 2)
    real_maxCH_info = torch.max(max_neighbor_ptps, dim = 1)
    real_maxCH_idx = real_maxCH_info[1]
    
    # print(real_maxCH_idx.get_device())
    # real_maxCH = col_idx[range(N), real_maxCH_idx.to('cpu')].long()
    real_maxCH = col_idx[range(N), real_maxCH_idx].long()
    
    wfs_denoised = wfs_denoised_mc_neighbors[range(N), real_maxCH_idx, :]
    
    
    
    
    DenoisedWF[range(N), : ,real_maxCH] = wfs_denoised # denoise all maxCH in one batch
    
    
    thresholds = torch.max(0.3*real_maxCH_info[0], torch.tensor(3)) # threshold to identify trustable neighboring channel
    
    mcs_phase_shift = torch.argmax(torch.abs(torch.squeeze(wfs_denoised)), 1) - offset
    spk_signs = torch.sign(wfs_denoised[range(N), np.ones(N)*offset])
    
    CH_checked = torch.zeros((N, C), device = device)
    CH_phase_shift = torch.zeros((N, C), dtype = torch.int64, device = device)
    parent = torch.full((N, C), float('nan'), device = device)


    parent_peak_phase = torch.zeros((N, C), device = device)

    CH_phase_shift[range(N), real_maxCH] = mcs_phase_shift

    wfs_ptp = torch.zeros((N, C), device = device)
    halluci_idx = torch.zeros((N, C), device = device).long()    

    # wfs_ptp[range(N), real_maxCH.to(device)] = max_neighbor_ptps[range(N), real_maxCH_idx]
    wfs_ptp[range(N), real_maxCH] = max_neighbor_ptps[range(N), real_maxCH_idx]
    
    real_maxCH = real_maxCH.detach().numpy()
    
    CH_checked[np.arange(N), real_maxCH] = 1
    
    Q = dict()
    for i in range(N):
        Q[i] = []
        Q[i].append(real_maxCH[i])
        
        
    manager = Manager()

    ci_graph_all = manager.dict()
    for i in range(N):
        


    while True:
        print(Q.values())
        if len(sum(Q.values(),[])) == 0:
            return DenoisedWF
        
        Q_neighbors = dict()
        for i in range(N):
            q = Q[i]
            if len(q)>0:
                ci_graph = ci_graph_all[i]

                u = q.pop()
                v = ci_graph[u]

                
                Q_neighbors[i] = list(v[CH_checked[i, v] == 0])
            else:
                Q_neighbors[i] = []
                
  
        #####
        for nodes in zip_longest(*(Q_neighbors[i] for i in range(N))):
            keep_N_idx = torch.tensor([j for j in range(len(nodes)) if nodes[j] is not None], device = device)
            for j in keep_N_idx:
                
                k = nodes[j].item()
                Q[j.item()].insert(0, k)
                    
                neighbors = ci_graph_all[j.item()][k]
                checked_neighbors = neighbors[CH_checked[j, neighbors] == 1]
                

                phase_shift_ref = torch.argmax(wfs_ptp[j,checked_neighbors])
                
                rest_phase_shift = torch.cat((parent_peak_phase[j,:checked_neighbors[phase_shift_ref]], parent_peak_phase[j,checked_neighbors[phase_shift_ref]+1:])).to(device)
                
                
                if (wfs_ptp[j, checked_neighbors[phase_shift_ref]] > thresholds[j]) & (torch.min(torch.abs(rest_phase_shift - CH_phase_shift[j, checked_neighbors[phase_shift_ref]]))<=5):
                    parent_peak_phase[j, k] = CH_phase_shift[j, checked_neighbors[phase_shift_ref]]
                else:
                    parent_peak_phase[j, k] = 0

                parent[j, k] = checked_neighbors[phase_shift_ref]
                CH_checked[j, k] = 1

            CH_idx = torch.tensor([nodes[j] for j in keep_N_idx], device = device)
            
            wfs = waveforms[keep_N_idx, : , CH_idx]

            spk_denoised_wfs, CH_phase_shift[keep_N_idx, CH_idx], halluci_idx[keep_N_idx, CH_idx] = denoise_with_phase_shift(wfs, parent_peak_phase[keep_N_idx, CH_idx], Denoiser, spk_signs[keep_N_idx])
                    
            
            DenoisedWF[keep_N_idx, : , CH_idx] = spk_denoised_wfs      

            wfs_ptp[keep_N_idx, CH_idx] = ptp(spk_denoised_wfs, 1)

        #####
        for i in range(N):
            v = Q_neighbors[i]
            if torch.sum(halluci_idx[i, v])>=3:
                q_partial = v#.tolist()
                while len(q_partial)>0:
                    x = q_partial.pop().item()
                    y = ci_graph_all[i][x]
                    for z in y:
                        if CH_checked[i, z] == 0:
                            CH_checked[i, z] = 1
                            q_partial.insert(0,z)
                            halluci_idx[i, z] = 1



