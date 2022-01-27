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
from scipy.signal import resample_poly


# creates Class-init, repr gives printing-info, frozen makes 'immutable'
# dot not use frozen here, bcs of inheritance to non-frozen RunRawData Class
@dataclass(init=True, repr=True, )  #frozen=True,
class RunInfo:
    '''Stores the details of one specific run to import'''
    sub: str  # patient id, e.g. '008'
    ses: str  # sessions id, e.g. 'EphysMedOn02'
    task: str  # task id, e.g. 'Rest'
    acq: str  # acquisition: stimulation and Dysk-Meds (StimOffLevo30)
    run: str  # run sequence, e.g. '01'
    sourcepath: str  # directory where source data (.Poly5) is stored
    bidspath: Any = None  # made after initiazing
    store_str: Any = None  # made after initiazing
    preproc_sett: str = None  # foldername for specific settings
    fig_path: str = None  # folder to store figures

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
        if not self.preproc_sett == None:
            if not os.path.exists(os.path.join(
                self.fig_path,
                'preprocessing',
                self.store_str,
                self.preproc_sett,
            )):
                os.mkdir(os.path.join(
                    self.fig_path,
                    'preprocessing',
                    self.store_str,
                    self.preproc_sett,
                ))


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



def resample(
    data, group, Fs_orig, Fs_new
):
    """
    Assuming downsampling; TO ADD DOC-STRING
    """
    data = data[group]
    down = int(Fs_orig / Fs_new)  # factor to down sample
    newdata = np.zeros((data.shape[0], data.shape[1],
                        int(data.shape[2] / down)))
    time = data[:, 0, :]  # all time rows from all windows
    newtime = time[:, ::down]  # all windows, only times on down-factor
    newdata[:, 0, :] = newtime  # alocate new times in new data array
    newdata[:, 1:, :] = resample_poly(
        data[:, 1:, :], up=1, down=down, axis=2
    )  # fill signals rows with signals

    return newdata


def save_arrays(
    data: dict, group: str, runInfo: Any,
):
    '''
    Function to save preprocessed 3d-arrays as npy-files.

    Arguments:
        - data (dict): containing 3d-arrays
        - group(str): group to save
        - runInfo (class): class containing info of spec-run

    Returns:
        - None
    '''
    # define (and make) directory
    f_dir = os.path.join(os.path.dirname(runInfo.fig_path),
                        'data/preprocessed',
                        runInfo.store_str)
    if not os.path.exists(f_dir):
        os.mkdir(f_dir)

    f_name = f'preproc_{runInfo.preproc_sett}_{group}.npy'
    np.save(os.path.join(f_dir, f_name), data[group])

