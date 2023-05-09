# %%
import numpy as np
import numpy.random as random
import os
import scipy
from scipy import signal
import math
from sklearn.decomposition import PCA
from .waveform_noise import kill_signal, search_noise_snippets, noise_whitener
try:
    import torch 
    from torch import nn, optim
    import torch.utils.data as Data
    from torch.nn import functional as F
    from torch import distributions
    HAVE_TORCH = True
except:
    HAVE_TORCH = False
from spike_psvae import denoise
from spike_psvae.denoise import SingleChanDenoiser
# %%
class TwoChanDenoiser(nn.Module):
    def __init__(self, pretrained_path=None, n_filters=[256, 128], filter_sizes=[5, 11], spike_size=121):
#             if torch.cuda.is_available():
#                 device = 'cuda:0'
#             else:
#                 device = 'cpu'
#             torch.cuda.set_device(device)
        assert(HAVE_TORCH)
        super(TwoChanDenoiser, self).__init__()
        feat1, feat2 = n_filters
        size1, size2 = filter_sizes
        #TODO For Loop 
        # self.conv1 = nn.Sequential(nn.Conv1d(2, feat1, size1), nn.ReLU()).to()
        # self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
        
        self.fully_connect1 = nn.Sequential(nn.Linear(2*spike_size, feat1, bias=False), nn.ReLU()).to()
        self.fully_connect2 = nn.Sequential(nn.Linear(feat1, feat2, bias=False), nn.ReLU())
        self.out = nn.Linear(feat2, spike_size)
        # n_input_feat = feat2 * spike_size
        # n_input_feat = feat2 * (spike_size - size1 - size2 + 2)
        # self.out = nn.Linear(n_input_feat, spike_size)
        self.pretrained_path = pretrained_path

    def forward(self, x):
        # x = x[:, None]\
        # print(x.shape)
        x = torch.reshape(x, (-1,2*121))
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        x = x.view(x.shape[0], -1)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = x.view(x.shape[0], -1)
        return self.out(x)

    #TODO: Put it outside the class
    def load(self):
        checkpoint = torch.load(self.pretrained_path, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self

    def train(self, fname_save, DenoTD, n_train=10000, n_test=500, EPOCH=2000, BATCH_SIZE=512, LR=0.0001):
        """
        DenoTD instance of Denoise Training Data class
        """
        print('Training NN denoiser')

        if os.path.exists(fname_save):
            return

        optimizer = torch.optim.Adam(self.parameters(), lr=LR)   # optimize all cnn parameters
        loss_func = nn.MSELoss()                       # the target label is not one-hotted

        wf_col_train, wf_clean_train = DenoTD.make_training_data(n_train)
        train = Data.TensorDataset(torch.FloatTensor(wf_col_train), torch.FloatTensor(wf_clean_train))
        train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

        wf_col_test, wf_clean_test = DenoTD.make_training_data(n_test)

        # training and testing
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                est = self(b_x)
                loss = loss_func(est, b_y[:,1,:])   # cross entropy loss b_y.cuda()
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                if step % 100 == 0:
                    est_test = self(torch.FloatTensor(wf_col_test))[0]
                    l2_loss = np.mean(np.square(est_test.cpu().data.numpy() - wf_clean_test[:, 1, :]))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| l2 loss: %.4f' % l2_loss)

        # save model
        torch.save(self.state_dict(), fname_save)
 

    
def iterative_denoise(wf, TwoChanDN, SingleChanDN, max_CH, geom, channel_index):
    n_wfs, n_times, n_channels = wf.shape
    wf = wf.transpose(0, 2, 1)
    
    # real_maxCH fromsingle channel denoised wfs
    wfs_denoised_old = SingleChanDN(torch.FloatTensor(wf).reshape(-1, 121)).reshape(wf.shape)
    wfs_denoised_old = wfs_denoised_old.detach().numpy()
    
    wfs_ptp = wfs_denoised_old.ptp(2)
    real_maxCH = np.argmax(wfs_ptp, axis = 1)

    denoised_all_wfs = []
    for i in range(n_wfs):
        # create a graph with all the neighboring channels connected
        wfs = wf[i, :, :]
        
        maxCH = max_CH[i] #max channel [0, 384]
        ci = channel_index[maxCH]
        
        x_pitch = np.diff(np.unique(geom[:,0]))[0]
        y_pitch = np.diff(np.unique(geom[:,1]))[0]
        
        # to do: define new function make_geom_graph
        not_nan_idx = (ci<384)
        ci_graph = dict()
        ci_geom = np.zeros((n_channels, 2))
        ci_geom[not_nan_idx] = geom[ci[not_nan_idx]]
        for ch in range(len(ci)):
            if not_nan_idx[ch]:
                ci_graph[ch] = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == y_pitch) & not_nan_idx)|
                                   ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * y_pitch) & not_nan_idx) |
                                   ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0) & not_nan_idx)) 
            else:
                ci_graph[ch] = []
            
        spk_denoised_wfs = np.zeros((n_channels, n_times))
        mcs_idx = real_maxCH[i]#np.squeeze(np.where(ci == maxCH))
        CH_checked = np.zeros(n_channels)
        wf_ptps = np.zeros(n_channels)
        
        mcs_wfs = wfs[mcs_idx,:]
    
        wfs_denoised = SingleChanDN(torch.FloatTensor(mcs_wfs).reshape(-1, 121)).reshape(mcs_wfs.shape)
        wfs_denoised = np.squeeze(wfs_denoised.detach().numpy())
        spk_denoised_wfs[mcs_idx, :] = wfs_denoised
        
        CH_checked[mcs_idx] = 1
        wf_ptps[mcs_idx] = wfs_denoised.ptp()
        
        q = []
        q.append(int(mcs_idx))
        parents = np.zeros(n_channels, int)

        parents[mcs_idx] = 0

        while len(q)>0:
            u = q.pop()
            v = ci_graph[u][0]
            np.random.shuffle(v) # randomly shuffle, is it necessary?

            for k in v:
                if CH_checked[k] == 0:
                    neighbors = ci_graph[k][0]
                    checked_neighbors = neighbors[CH_checked[neighbors] == 1]

                    CH_ref = checked_neighbors[np.argmax(wf_ptps[checked_neighbors])]
                    
                    Two_CH_wfs = np.concatenate((spk_denoised_wfs[np.int32(CH_ref),:][None,:], wfs[np.int32(k),:][None,:]), axis = 0)
                    
                    wfs_denoised = TwoChanDN(torch.FloatTensor(Two_CH_wfs[None, :, :]))
                    wfs_denoised = np.squeeze(wfs_denoised.detach().numpy())
                    spk_denoised_wfs[k,:] = wfs_denoised
                    wf_ptps[k] = wfs_denoised.ptp()

                    parents[k] = CH_ref

                    q.insert(0,k)
                    CH_checked[k] = 1
                    
        denoised_all_wfs.append(spk_denoised_wfs[None,:,:])
    return np.array(denoised_all_wfs)

def wfs_similarity(wf1, wf2):
    #The cosine distance between waveforms on two channels. Has a value betwen [-1, 1]
        return np.sum(wf1 * wf2, axis = 1)/(np.linalg.norm(wf1, ord = 2, axis = 1)*np.linalg.norm(wf2, ord = 2, axis = 1))

# %%
class Denoising_Training_Data_Two_Channels(object):
    """
    create training dataset with one denoised neighboring channel and the noisy channel to be denoised
    templates obtained by simple clustering + averaging waveforms + crop and align
    spatial_sig, temporal_sig obtained from spikeinterface.sortingcomponents.waveform_tools.noise_whitener
    """

    def __init__(self,
                 templates, #templates with 384 channels as input, return two channel templates as training data.
                 max_chan,
                 channel_index,
                 spatial_sig,
                 temporal_sig,
                 geom_array):

        self.spatial_sig = spatial_sig
        self.temporal_sig = temporal_sig
        self.templates = templates
        self.channel_index = channel_index
        self.spike_size = self.temporal_sig.shape[0]
        self.geom = geom_array
        self.max_chan = max_chan
        self.x_pitch = np.diff(np.unique(self.geom[:,0]))[0]
        self.y_pitch = np.diff(np.unique(self.geom[:,1]))[0]

        self.remove_small_templates()
        #remove templates that has too small maxchan ptp
        
        
        self.create_two_channel_templates()
        self.standardize_templates()
        print(np.shape(self.two_channel_templates))
        self.augment_two_channel_templates()
        # self.jitter_templates()



    def remove_small_templates(self):
        ptp = np.max(self.templates.ptp(1),axis = 1)
        self.templates = self.templates[ptp > 3,:,:]
        
    def create_two_channel_templates(self):
        n_templates, n_times, n_channels = self.templates.shape
        
        two_channel_templates = []
        
        for i in range(n_templates):
            
            # create a graph with all the neighboring channels connected
            maxCH = self.max_chan[i] #max channel [0, 384]
            ci = self.channel_index[maxCH]
            
            wfs_ptp = self.templates[i,:,:].ptp(0)
            
            
            not_nan_idx = (ci<384)
            ci_graph = dict()
            ci_geom = np.zeros((len(ci), 2))
            ci_geom[not_nan_idx,:] = self.geom[ci[not_nan_idx],:]
            for ch in range(len(ci)):
                if not_nan_idx[ch]:
                    ci_graph[ch] = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == self.x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == self.y_pitch) & not_nan_idx)|
                                       ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * self.y_pitch) & not_nan_idx) |
                                       ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * self.x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0) & not_nan_idx))
                else:
                    ci_graph[ch] = []
            
            mcs_idx = np.squeeze(np.where(ci == maxCH))
            CH_checked = np.zeros(len(ci))

            CH_checked[mcs_idx] = 1
            q = []
            q.append(int(mcs_idx))
            parents = np.zeros(n_channels, int)

            parents[mcs_idx] = 0
            
            full_temp = self.templates[i,:,:]

            while len(q)>0:
                u = q.pop()
                v = ci_graph[u][0]
                # v = np.random.shuffle(v) # randomly shuffle, is it necessary?
                for k in v:
                    if CH_checked[k] == 0:
                        neighbors = ci_graph[k][0]
                        checked_neighbors = neighbors[CH_checked[neighbors] == 1]
                        
                        CH_ref = np.argmax(wfs_ptp[ci[checked_neighbors]])

                        parents[k] = checked_neighbors[CH_ref]

                        q.insert(0,k)
                        CH_checked[k] = 1
                        
                        if (wfs_ptp[ci[parents[k]]] > 3) & (wfs_ptp[ci[k]] > 3):
                            two_channel_templates.append(full_temp[:, [ci[parents[k]], ci[k]]])
             
        self.two_channel_templates = np.array(two_channel_templates)
        
        

    def standardize_templates(self):

        # standardize templates
        ptp = np.abs(self.two_channel_templates[:,:,1]).max(1)
        self.two_channel_templates = self.two_channel_templates/ptp[:,None, None]


    def jitter_templates(self, up_factor=8):

        n_templates, n_times, n_channel = self.two_channel_templates.shape

        # upsample best fit template
        up_temp = scipy.signal.resample(
            x=self.two_channel_templates[:,:,1],
            num=n_times*up_factor,
            axis=1)
        up_temp = up_temp.T

        idx = (np.arange(0, n_times)[:,None]*up_factor + np.arange(up_factor))
        up_shifted_temps = up_temp[idx].transpose(2,0,1)
        up_shifted_temps = np.concatenate(
            (up_shifted_temps,
             np.roll(up_shifted_temps, shift=1, axis=1)),
            axis=2)
        up_shifted_temps = up_shifted_temps.transpose(0,2,1).reshape(-1, n_times)

        ref = np.mean(up_shifted_temps, 0)
        
        print(np.shape(up_shifted_temps))
        shifts=  align_get_shifts_with_ref(
            up_shifted_temps, ref, upsample_factor=1)
        self.two_channel_templates = shift_chans(self.two_channel_templates, shifts)


    def augment_two_channel_templates(self):
        CH_wf_dist = wfs_similarity(self.two_channel_templates[:,:,0], self.two_channel_templates[:,:,1])
        # augment templates that have different wfs on two channels 
        bins = np.arange(-0.5, 1, 0.1)
        dist_digitize = np.digitize(CH_wf_dist, bins)
        
        c, b = np.histogram(CH_wf_dist, bins=bins)
        
        bin_n = np.max(c)
        # print(bin_n)
        augment_id = []
        
        inbin_idx = np.where(dist_digitize == 0)[0]
        if (len(inbin_idx)>0) & (len(inbin_idx)<bin_n):
            augment_id.append(np.random.choice(inbin_idx, bin_n))
        
        for i in range(len(bins)-1):
            inbin_idx = np.where(dist_digitize == (i + 1))[0]
            if (len(inbin_idx)>0) & (len(inbin_idx)<bin_n):
                augment_id.append(np.random.choice(inbin_idx, bin_n))
            elif len(inbin_idx)>=bin_n:
                augment_id.append(inbin_idx)
        augment_id = np.array(augment_id).flatten()
        total_n = len(augment_id)
        print(total_n)
        
#         # augment positive spikes to 10%
#         wfs2 = self.two_channel_templates[:,:,1]
#         positive_idx = np.where(np.max(np.abs(wfs2), 1) == np.max(wfs2, 1))[0]
#         augment_id = np.concatenate((augment_id[:, None],np.random.choice(positive_idx, int(total_n*0.15))[:, None]))
#         print('positive size:' + str(len(positive_idx)))
        
#         # augment templates that have bigger second channel to 10%
#         ptp1 = self.two_channel_templates[:,:,0].ptp(1)
#         ptp2 = self.two_channel_templates[:,:,1].ptp(1)
#         second_big_id = np.where(ptp2>1.5*ptp1)[0]
#         augment_id = np.concatenate((augment_id,np.random.choice(second_big_id, int(total_n*0.15))[:, None]))
# #         print('big second channel size:' + str(len(second_big_id)))
        
# #         # augment templates that travels to 10%
#         wf1 = self.two_channel_templates[:,:,0]
#         wf1_peak = np.argmax(np.abs(wf1), axis = 1)
#         wfs1_peak_sign = np.sign(wf1[np.arange(len(wf1_peak)), wf1_peak])
        
#         wf2 = self.two_channel_templates[:,:,1]
#         wf2_peak = np.argmax(wf2 * wfs1_peak_sign[:, None], axis = 1)
        
#         travel_idx = np.where(np.abs(wf1_peak - wf2_peak)>3)[0]
#         augment_id = np.concatenate((augment_id,np.random.choice(travel_idx, int(total_n*0.15))[:, None]))
        
# # #         print('travel size:' + str(len(travel_idx)))

#         augment_id = np.array(augment_id)
        
#         print(np.shape(augment_id))
        augment_id = np.sort(augment_id)
        
        self.two_channel_templates = self.two_channel_templates[np.squeeze(augment_id), :, :]
    
    
    
    def make_training_data(self, n):

        n_templates, n_times, n_channels = self.two_channel_templates.shape

        center = n_times//2
        t_idx_in = slice(center - self.spike_size//2,
                         center + (self.spike_size//2) + 1)

        # sample templates
        idx1 = np.random.choice(n_templates, n)
        idx2 = np.random.choice(n_templates, n)
        wf1 = self.two_channel_templates[idx1]
        wf2 = self.two_channel_templates[idx2]

        # sample scale
        # s1 = np.exp(np.random.randn(n)*0.8 + 2)
        # s2 = np.exp(np.random.randn(n)*0.8 + 2)
        s1 = np.exp(np.random.randn(n)*0.8 + 2)
        s2 = np.exp(np.random.randn(n)*0.8 + 2)
        
        # swap two channels, not really changing anything
        wf1 = np.random.permutation(wf1.transpose(2,0,1)).transpose(1,2,0)
        wf2 = np.random.permutation(wf2.transpose(2,0,1)).transpose(1,2,0)

        # turn off some
        c1 = np.random.binomial(1, 1-0.10, n) # probability of turining off only the second channel
        c11 = np.random.binomial(1, 1-0.05, n) # probability of turning off both channels
        c2 = np.random.binomial(1, 1-0.05, n)

        # multiply them
        wf1[:,:,1] = wf1[:,:,1]*c1[:, None]
        wf1 = wf1*s1[:, None, None]*c11[:, None, None]
        wf2 = wf2*s2[:, None, None]*c2[:, None, None]

        # choose shift amount
        shift = np.random.randint(low=0, high=3, size=(n,))
        sub_shift = np.random.randint(low=0, high=3, size=(n,))

        # choose shift amount
        # shift21 = np.random.randint(low=5, high=self.spike_size, size=(int(n*0.8),))
        # shift22 = np.random.randint(low=4, high=6, size=(n - int(n*0.8),))
        # shift2 = np.squeeze(np.concatenate((shift21[:, None],shift22[:, None])))
        shift2 = np.random.randint(low=5, high=self.spike_size, size=(n,))

        shift *= np.random.choice([-1, 1], size=n)
        sub_shift *= np.random.choice([-1, 1], size=n)
        shift2 *= np.random.choice([-1, 1], size=n, p=[0.2, 0.8])

        # make colliding wf, only for the second channel. Do we also include the max channel waveform as collision to avoid bias? Something to check later
        wf_clean = np.zeros(wf1.shape)
        for j in range(n):
            temp = np.roll(wf1[j,:,:], shift[j], axis = 0)
            # temp[:,1] = np.roll(temp[:,1], sub_shift[j], axis = 0)
            wf_clean[j] = temp

        # make colliding wf    
        wf_col = np.zeros(wf2.shape)
        for j in range(n):
            temp = np.roll(wf2[j,:,:], shift2[j], axis = 0)
            wf_col[j] = temp
            
        wf_col = np.concatenate((np.zeros((n, n_times, 1)), wf_col[:,:,1][:,:,np.newaxis]), axis = 2)

        noise_wf = np.concatenate((np.zeros((n, n_times, 1)), make_noise(n, self.spatial_sig, self.temporal_sig)[:, :, 0][:,:,np.newaxis]), axis = 2) # add noise only to the second channel

        wf_clean = wf_clean[:, t_idx_in,:]
        wf_noisy = (wf_clean + wf_col[:, t_idx_in, :] + noise_wf).transpose(0, 2, 1)
        wf_clean = wf_clean.transpose(0, 2, 1)
        return (wf_noisy,
                wf_clean)
    
    
# # %%
# pretrained_path = (
#     Path(__file__).parent.parent / "pretrained/single_chan_denoiser.pt"
# )


# class SingleChanDenoiser(nn.Module):
#     """Cleaned up a little. Why is conv3 here and commented out in forward?"""

#     def __init__(
#         # self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121
#         self, n_filters=[16, 8], filter_sizes=[5, 11], spike_size=121
#     ):
#         super(SingleChanDenoiser, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv1d(1, n_filters[0], filter_sizes[0]), nn.ReLU())
#         self.conv2 = nn.Sequential(nn.Conv1d(n_filters[0], n_filters[1], filter_sizes[1]), nn.ReLU())
#         if len(n_filters) > 2:
#             self.conv3 = nn.Sequential(nn.Conv1d(n_filters[1], n_filters[2], filter_sizes[2]), nn.ReLU())
#         n_input_feat = n_filters[1] * (spike_size - filter_sizes[0] - filter_sizes[1] + 2)
#         self.out = nn.Linear(n_input_feat, spike_size)

#     def forward(self, x):
#         x = x[:, None]
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # x = self.conv3(x)
#         x = x.view(x.shape[0], -1)
#         return self.out(x)

#     def load(self, fname_model=pretrained_path):
#         checkpoint = torch.load(fname_model, map_location="cpu")
#         self.load_state_dict(checkpoint)
#         return self








# def denoise_wf_nn_two_channels(wf, denoiser, device):
#     """
#     This function NN-denoises waveform arrays 
#     TODO: avoid sending back and forth
#     """
#     assert(HAVE_TORCH)
#     denoiser = denoiser.to(device)
#     n_data, n_times, n_chans = wf.shape
#     if wf.shape[0] > 0:
#         # wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
#         # wf_torch = torch.FloatTensor(wf_reshaped).to(device)
#         # denoised_wf = denoiser(wf_torch).data
#         # denoised_wf = denoised_wf.reshape(n_data, n_chans, n_times)
#         # denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)
        
        

#         del wf_torch
#     else:
#         denoised_wf = np.zeros_like(wf)

#     return denoised_wf


# %%
def load_nn_and_denoise(wf_array, denoiser_weights_path, architecture_path):
    """
    This function Load NN and NN-denoises waveform arrays 
    TODO: Delete this function?
    """

    assert(HAVE_TORCH)
    architecture_denoiser = np.load(architecture_path, allow_pickle = True)

    model = SingleChanDenoiser(denoiser_weights_path,
                    architecture_denoiser['n_filters'], 
                    architecture_denoiser['filter_sizes'], 
                    architecture_denoiser['spike_size'])
    denoiser = model.load()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    return denoise_wf_nn_single_channel(wf_array, denoiser, device)

# %%
def shift_chans(wf, best_shifts):
    ''' 
    Align all waveforms on a single channel given shifts
    '''
    # use template feat_channel shifts to interpolate shift of all spikes on all other chans
    wfs_final= np.zeros(wf.shape, 'float32')
    for k, shift_ in enumerate(best_shifts):
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(wf[k],ceil,axis=0)
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            temp = np.roll(wf[k],ceil,axis=0)*(shift_-floor)+np.roll(wf[k],floor, axis=0)*(ceil-shift_)
        wfs_final[k] = temp
    
    return wfs_final

# %%
def align_get_shifts_with_ref(wf, ref=None, upsample_factor=5, nshifts=7):

    ''' Returns shifts for aligning all waveforms on a single channel (ref)
    
        Used to generate training data
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    n_data, n_time = wf.shape

    if ref is None:
        ref = np.mean(wf, axis=0)
      
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1

    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = nshifts//2
    wf_end = -nshifts//2
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-(nshifts//2), (nshifts//2)+1)):
        ref_shifted[:,i] = ref_upsampled[s + wf_start: s + wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts/np.float32(upsample_factor)

# %%
def upsample_resample(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces

# %%
def make_noise(n, spatial_SIG, temporal_SIG):
    """Make noise
    Parameters
    ----------
    n: int
        Number of noise events to generate
    Returns
    ------
    numpy.ndarray
        Noise
    """
    n_neigh, _ = spatial_SIG.shape
    waveform_length, _ = temporal_SIG.shape

    # get noise
    noise = np.random.normal(size=(n, waveform_length, n_neigh))

    for c in range(n_neigh):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)
        reshaped_noise = np.reshape(noise, (-1, n_neigh))

    the_noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                           (n, waveform_length, n_neigh))

    return the_noise





