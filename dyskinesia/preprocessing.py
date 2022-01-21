# Functions to pre-process neurophysiology data (LFP and ECOG)
# in CFB295 ReTune's Dyskinesia Project

# Importing general packages and functions
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import matplotlib.pyplot as plt
import mne_bids
import mne

# DATACLASSES TO SPECIFY, STORE, AND LOAD PATIENT-RUN INFO


# creates Class-init, repr gives printing-info, frozen makes 'immutable'
# dot not use frozen here, bcs of inheritance to non-frozen RunRawData Class
@dataclass(init=True, repr=True, )  #frozen=True,
class RunInfo:
    '''Stores the details of one specific run to import'''
    # check: ??? @abstractmethod combi dataclass, metaclass and absrtacmethod
    sub: str  # patient id, e.g. '008'
    ses: str  # sessions id, e.g. 'EphysMedOn02'
    task: str  # task id, e.g. 'Rest'
    acq: str  # acquisition, often stimulation, e.g. 'StimOff'
    # in dyskinesia experiments levodopa status is added: 'StimOffLD00'
    run: str  # run sequence, e.g. '01'
    sourcepath: str  # directory where source data (.Poly5) is stored
    bidspath: Any = None  # is made after initiazing based on other values
    store_str: Any = None  # is made after initiazing based on other values

    def __post_init__(self,):  # is called after initialization
        bidspath = mne_bids.BIDSPath(
            subject=self.sub,
            session=self.ses,
            task=self.task,
            acquisition=self.acq,
            run=self.run,
            suffix='ieeg',
            extension='.vhdr',
            datatype='ieeg',
            root=self.sourcepath,
        )
        store_str = (f'{self.sub}_{self.ses}_{self.task}_'
                     f'{self.acq}_{self.run}')
        # bidspath for MNE functions
        self.bidspath = bidspath
        # folder name to store figures and derivative
        self.store_str = store_str

# change value via object.__setattr__ if frozen=True
# object.__setattr__(self, 'bidspath', bidspath)
# consider using 'slots'=variables in Class
# __slots__ = [list of var's]
# this decreases the memory and increases the speed of the Class


@dataclass(init=True, repr=True,)
class RunRawData:
      '''Collect data from BIDS and stores them per data type.
      Actial loading in of the data happens later.'''
      bidspath: Any  # takes RunIno.bidspath as input kw
      bids: Any = None
      lfp: Any = None
      lfp_left: Any = None
      lfp_right: Any = None
      ecog: Any = None
      acc: Any = None
      emg: Any = None
      ecg: Any = None


      def __post_init__(self, ):
      # read raw bids files into mne RawBrainVision object
      # doc mne.io.raw: https://mne.tools/stable/generated/mne.io.Raw.html
            self.bids = mne_bids.read_raw_bids(self.bidspath, verbose='WARNING')
            # print bids-info to check
            print('\n\n------------ BIDS DATA INFO ------------\n'
                  f'The raw-bids-object contains {len(self.bids.ch_names)} channels with '
                  f'{self.bids.n_times} datapoints and sample freq ',
                  str(self.bids.info['sfreq']),'Hz')
            print('Bad channels are:',self.bids.info['bads'],'\n')

            # select ECOG vs DBS channels (bad channels are dropped!)
            self.ecog = self.bids.copy().pick_types(ecog=True, exclude='bads')
            self.lfp = self.bids.copy().pick_types(dbs=True, exclude='bads')
            # accelerometer channels are coded as misc(ellaneous)
            self.acc = self.bids.copy().pick_types(misc=True, exclude='bads')
            # splitting LFP in left right
            lfp_chs_L = [c for c in self.lfp.ch_names if c[4]=='L' ]
            lfp_chs_R = [c for c in self.lfp.ch_names if c[4]=='R' ]
            self.lfp_left = self.lfp.copy().drop_channels(lfp_chs_R)
            self.lfp_right = self.lfp.copy().drop_channels(lfp_chs_L)
            # select EMG and ECG signals
            self.emg = self.bids.copy().pick_types(emg=True, exclude='bads')
            self.ecg = self.bids.copy().pick_types(ecg=True, exclude='bads')

            print(f'BIDS contains:\n{len(self.ecog.ch_names)} ECOG '
                  f'channels,\n{len(self.lfp.ch_names)} DBS channels:'
                  f' ({len(self.lfp_left.ch_names)} left, '
                  f'{len(self.lfp_right.ch_names)} right), '
                  f'\n{len(self.emg.ch_names)} EMG channels, '
                  f'\n{len(self.ecg.ch_names)} ECG channel(s), '
                  f'\n{len(self.acc.ch_names)} Accelerometry (misc) channels.\n\n')
        
# for detailed exploraiton of content bids-obejcts:
# e.g. run1.bids.info.keys(), or dir(run1.bids)


# FUNCTIONS FOR PREPROCESSING

def block_artefact_selection(
    bids_dict: dict,
    group: str,
    win_len: float=.5,
    overlap=None,
    n_stds_cut: float=2.5,
    save=None
):
    '''Function to perform pre-processing, including visualization
    of raw data for artefacrt selection and deletion.
    Checks per block whether there is an outlier (value higher or low
    than n_std_cut times std dev of full recording). If an outlier
    present: full block is reverted to missing (np.nan). OR if
    more than 25% is exactly 0 in a block -> block is set to nan.

    CHECK: artefact deletion based on values or blocks?
    
    Input:
    - bids_dict, Raw BIDS selection: grouped BIDS raw, e.g. rawRun1.ecog,
    - win_len (float): block window length in seconds,
    - overlap (float): time of overlap between consec blocks (seconds),
    - n_stds_cut, int: number of std-dev's above and below mean that
        is used for cut-off's in artefact detection,
    - save (str): 1) directory where to store figure, 2) 'show' to only
        plot in notebook, 3) None to not plot.
    Output:
    sel_bids (array): array with all channels in which artefacts
        are replaced by np.nan's.
    '''
    print(f'START ARTEFACT REMOVAL: {group}')
    data = bids_dict[group]
    ch_nms = data.ch_names
    fs = data.info['sfreq']  # ONLY FOR BLOCKS
    (ch_arr, ch_t) = data.get_data(return_times=True)
    # visual check by plotting before selection
    if save:
        fig, axes = plt.subplots(len(ch_arr), 2, figsize=(16, 16))
        for n, c in enumerate(np.arange(len(ch_arr))):
            axes[c, 0].plot(ch_t, ch_arr[c, :])
            axes[c, 0].set_ylabel(ch_nms[n], rotation=90)
        axes[0, 0].set_title('Raw signal BEFORE artefact deletion')

    # Artefact removal part
    win_n = int(win_len * fs)  # number of samples to fit in one window
    n_wins = int(ch_arr.shape[1] / win_n)  # num of windows to split in
    # new array to store data without artefact, ch + 1 is for time
    new_arr = np.zeros((n_wins, len(ch_nms) + 1, win_n), dtype=float)
    n_nan = {}  # number of blocks corrected to nan
    # first reorganize data
    for w in np.arange(new_arr.shape[0]):  # loop over new window's
        # first row of windows is time
        new_arr[w, 0, :] = ch_t[w * win_n:w * win_n + win_n]
        # other rows filled with channels
        new_arr[w, 1:, :] = ch_arr[:, w * win_n:w * win_n + win_n]
    # correct to nan for artefacts per channel
    cuts = {}  # to store thresholds per channel
    for c in np.arange(ch_arr.shape[0]):  # loop over ch-rows
        # cut-off's are X std dev above and below channel-mean
        cuts[c] = (np.mean(ch_arr[c]) - (n_stds_cut * np.std(ch_arr[c])),
                np.mean(ch_arr[c]) + (n_stds_cut * np.std(ch_arr[c])))
        n_nan[c] = 0
        for w in np.arange(new_arr.shape[0]):  # loop over windows
            if (new_arr[w, c + 1, :] < cuts[c][0]).any():
                new_arr[w, c + 1, :] = [np.nan] * win_n
                n_nan[c] = n_nan[c] + 1
            elif (new_arr[w, c + 1, :] > cuts[c][1]).any():
                new_arr[w, c + 1, :] = [np.nan] * win_n
                n_nan[c] = n_nan[c] + 1
            elif (new_arr[w, c + 1, :] == 0).sum() > (.25 * win_n):
                # more than 25% exactly 0
                new_arr[w, c + 1, :] = [np.nan] * win_n
                n_nan[c] = n_nan[c] + 1

    # visual check by plotting after selection
    if save:
        for c in np.arange(len(ch_arr)):
            plot_ch = []
            for w in np.arange(new_arr.shape[0]):
                plot_ch.extend(new_arr[w, c + 1, :])  # c + 1 to skip time
            plot_t = ch_t[:len(plot_ch)]
            axes[c, 1].plot(plot_t, plot_ch, color='blue')
            # color shade isnan parts
            ynan = np.isnan(plot_ch)
            ynan = ynan * 2
            axes[c, 1].fill_between(
                x=plot_t,
                y1=cuts[c][0],
                y2=cuts[c][1],
                color='red',
                alpha=.3,
                where=ynan > 1,
            )
            axes[c, 1].set_title(f'{n_nan[c]} windows deleted')
        fig.suptitle('Raw signal artefact deletion (cut off: '
                f'{n_stds_cut} std dev +/- channel mean', size=14)
        fig.tight_layout(h_pad=.2)

        if save != 'show':
            try:
                f_name = (f'{group}_artefact_blockremoval_'
                        f'cutoff_{n_stds_cut}_sd.jpg')
                plt.savefig(os.path.join(save, f_name), dpi=150,
                            facecolor='white')
                plt.close()
            except FileNotFoundError:
                print(f'Directory {save} is not valid')
        elif save == 'show':
            plt.show()
            plt.close()

    ## DROP CHANNELS WITH TOO MANY NAN'S
    ch_sel = [0, ]  # channels to keep, incl time
    ch_nms_out = ['time', ]
    for c in np.arange(new_arr.shape[1]):  # loop over rows of windows
        if c == 0:
            continue  # skip time row
        else:
            nans = np.isnan(new_arr[:, c, :]).sum()
            length = new_arr.shape[0] * new_arr.shape[2]
            nanpart = nans / length
            print(f'Ch {c}: {round(nanpart * 100, 2)}'
                  f'% is NaN (artefact or zero)')
            if nanpart < .5:
                ch_sel.append(c)
                ch_nms_out.append(ch_nms[c - 1])
    # EXCLUDE BAD CHANNELS WITH TOO MANY Nan's and Zero's
    out_arr = new_arr[:, ch_sel, :]


    return out_arr, ch_nms_out


def block_bp_filter(
    clean_dict, group, sfreq, l_freq, h_freq,
    method='fir', fir_window='hamming', verbose=False
):
    '''
    DOCT STRING TO WRITE
    '''
    data_out = clean_dict[group].copy()
    for w in np.arange(clean_dict[group].shape[0]):
        data_out[w, 1:, :] = mne.filter.filter_data(
            data=clean_dict[group][w, 1:, :],
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            fir_window=fir_window,
            verbose=verbose,
        )

    return data_out


def block_notch_filter(
    bp_dict: dict,
    group: str,
    transBW: int,  # Default in doc is 1
    notchW: int,  # Deafult in doc is f/200
    Fs: int = 4000,  # sample freq, default 4000 Hz
    freqs: list = [50, 100, 150],  # freq's to filter (EU powerline 50 Hz)
    save=None,
    verbose='Warning'
):
    '''
    Applies notch-filter to filter local peaks due to powerline
    noise. Uses mne-fucntion (see doc).
    Inputs:
    - data (array): 2D data array with channels to filter,
    - transBW (int): transition bandwidth, try out and decide on
        defaults for LFP and ECOG seperately,
    - notchW (int): notch width, try out and decide on
        defaults for LFP and ECOG seperately,
    - save (str): if pre and post-filter figures should be saved,
        directory should be given here,
    - verbose (str): amount of documentation printed.
    Output:
    - data: filtered data array.
    '''
    data = getattr(bp_dict, group)
    data_out = data.copy()
    if save:
        plot_wins = np.arange(1000, 1020)  # select range of win's to plot
        # visualize before notch filter
        fig, axes = plt.subplots(data.shape[1] - 1, 2, figsize=(12, 12))
        for C in np.arange(1, data.shape[1]):
            for w in plot_wins:
                axes[C-1, 0].psd(data[w, C, :], Fs=4000,
                               NFFT=1024, label=str(w))
                axes[C-1, 0].set_xlim(0, 160)
        axes[0, 0].set_title('PSD before NOTCH filter')

    # apply notch filter
    for w in np.arange(data.shape[0]):
        data_out[w, 1:, :] = mne.filter.notch_filter(
            x=data[w, 1:, :],
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
        # visualize after notch filter
        for C in np.arange(1, data.shape[1]):
            for w in plot_wins:
                axes[C-1, 1].psd(data[w, C, :], Fs=4000,
                               NFFT=1024, label=str(w))
                axes[C-1, 1].set_xlim(0, 160)
        axes[0, 1].set_title('PSD AFTER NOTCH filter')

        fname = f'{group}_Notch_block_transBW{transBW}_notchW{notchW}.jpg'
        plt.savefig(os.path.join(save, fname), dpi=150,
                    faceccolor='white')
        plt.close()
    
    return data