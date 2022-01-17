# Functions to pre-process neurophysiology data (LFP and ECOG)
# in CFB295 ReTune's Dyskinesia Project

# Importing general packages and functions
import os
import numpy as np
import matplotlib.pyplot as plt
import mne_bids
import mne


def artefact_selection(sel_bids, n_stds_cut=2.5, save=None):
    '''Function to perform pre-processing, including visualization
    of raw data for artefacrt selection and deletion.

    CHECK: artefact deletion based on values or blocks?
    
    Input:
    sel_bids, Raw BIDS selection: grouped BIDS raw, e.g. rawRun1.ecog,
    n_stds_cut, int: number of std-dev's above and below mean that
        is used for cut-off's in artefact detection;
    save_dir (str): directory where to store figure, it not inserted
        default=None, there is no figure saved.
    Output:
    sel_bids (array): array with all channels in which artefacts
        are replaced by np.nan's.
    '''
    ch_nms = sel_bids.ch_names
    fs = sel_bids.info['sfreq']  # ONLY FOR BLOCKS
    (ch_arr, ch_t) = sel_bids.get_data(return_times=True)
    # visual check by plotting before selection
    fig, axes = plt.subplots(len(ch_arr), 2, figsize=(16, 16))
    for n, c in enumerate(np.arange(len(ch_arr))):
        axes[c, 0].plot(ch_arr[c, :])
        axes[c, 0].set_ylabel(ch_nms[n], rotation=90)
    axes[0, 0].set_title('Raw signal BEFORE artefact deletion')
    # win_n = win_len / (1 / fs)  # number of samples to fit in one window  # ONLY FOR BLOCKS
    n_del = {}  # to store number of points deleted
    for c in np.arange(ch_arr.shape[0]):
        # cut-off's are X std dev above and below channel-mean
        cuts = [np.mean(ch_arr[c]) - (n_stds_cut * np.std(ch_arr[c])),
                np.mean(ch_arr[c]) + (n_stds_cut * np.std(ch_arr[c]))]
        n_del[c] = 0
        for i in np.arange(len(ch_arr[c])):
            if ch_arr[c][i] < cuts[0]:
                ch_arr[c][i] = np.nan
                n_del[c] = n_del[c] + 1
            elif ch_arr[c][i] > cuts[1]:
                ch_arr[c][i] = np.nan
                n_del[c] = n_del[c] + 1
        print(f'In Channel {c}: deleted {n_del[c]} data points.')

    # visual check by plotting after selection
    for n, c in enumerate(np.arange(len(ch_arr))):
        axes[c, 1].plot(ch_arr[c, :])
        axes[c, 1].set_title(f'{n_del[c]} data points deleted')
    fig.suptitle('Raw signal artefact deletion (cut off: '
              f'{n_stds_cut} std dev +/- channel mean', size=14)
    fig.tight_layout(h_pad=.2)

    if save:
        if not os.path.isdir(save):
            os.mkdir(save)
        f_name = f'artefact_removal_cutoff_{n_stds_cut}_sd.jpg'
        plt.savefig(os.path.join(save, f_name), dpi=150,
                    facecolor='white')
    plt.close()

    return ch_arr



def notch_filter(
    data,
    transBW=1,  # Default in doc is 1
    notchW=1,  # Deafult in doc is f/200
    Fs=4000,  # sample freq, default 4000 Hz
    freqs=[50, 100],  # freq's to filter (EU powerline 50 Hz)
    save=None,
    verbose='Warning'
):
    '''
    Applies notch-filter to filter local peaks due to powerline
    noise. Uses mne-fucntion (see doc).
    Inputs:
    - data (array): 2D data array with channels to filter,
    - transBW (int): transition bandwidth,
    - notchW (int): notch width,
    - save (str): if pre and post-filter figures should be saved,
        directory should be given here,
    - verbose (str): amount of documentation printed.
    Output:
    - data: filtered data array.
    '''
    if save:
        fig, axes = plt.subplots(6, 2, figsize=(12, 12))
        for C in np.arange(data.shape[0]):
            for N in np.arange(0, 10):
                axes[C, 0].psd(data[C, (0 * N):(8000 * N)],
                                Fs=4000, NFFT=1024, label=str(N))
                axes[C, 0].set_xlim(0, 160)
        axes[0, 0].set_title('PSD before NOTCH filter')

    for r in np.arange(data.shape[0]):
        data[r, :] = mne.filter.notch_filter(
            x=data[r, :],
            Fs=Fs,
            freqs=freqs,
            trans_bandwidth=transBW,
            notch_widths=notchW,
            method='fir',
            fir_window='hamming',
            fir_design='firwin',
            verbose=verbose,
        )

    if save:
        for C in np.arange(data.shape[0]):
            for N in np.arange(0, 10):
                axes[C, 1].psd(data[C, (0 * N):(8000 * N)],
                                Fs=4000, NFFT=1024, label=str(N))
                axes[C, 1].set_xlim(0, 160)
        axes[0, 1].set_title('PSD AFTER NOTCH filter')

        fname = f'Notch_transBW{transBW}_notchW{notchW}.jpg'
        plt.savefig(os.path.join(save, fname), dpi=150,
                    faceccolor='white')
        plt.close()
    
    return data
