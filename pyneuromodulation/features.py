#from multiprocessing import Process, Manager
from numpy import arange, array, ceil, floor

from mne.filter import notch_filter

import bandpower
import filter
import hjorth_raw
import kalmanfilter
import sharpwaves


class Features:
    
    def __init__(self, s, fs, line_noise, channels) -> None:
        """
        s (dict) : json settings
        """

        self.ch_names = channels
        self.s = s  # settings
        self.fs = fs
        self.line_noise = line_noise
        self.seglengths = floor(
                self.fs / 1000 * array([value[1] for value in s[
                    "bandpass_filter_settings"][
                        "frequency_ranges"].values()])).astype(int)
        print("segment lengths:", self.seglengths)
        self.KF_dict = {}
        
        if s["methods"]["bandpass_filter"] is True:
            self.filter_fun = filter.calc_band_filters(f_ranges=[
                value[0] for value in s["bandpass_filter_settings"][
                    "frequency_ranges"].values()], sfreq=fs,
                filter_length=fs - 1)
        
        if s["methods"]["kalman_filter"] is True:
            for bp_feature in [k for k, v in s["bandpass_filter_settings"][
                               "bandpower_features"].items() if v is True]:
                for f_band in s["kalman_filter_settings"]["frequency_bands"]:
                    for channel in self.ch_names:
                        self.KF_dict['_'.join([channel, bp_feature, f_band])] \
                            = kalmanfilter.define_KF(
                            s["kalman_filter_settings"]["Tp"],
                            s["kalman_filter_settings"]["sigma_w"],
                            s["kalman_filter_settings"]["sigma_v"])
        
        if s["methods"]["sharpwave_analysis"] is True:
            self.sw_features = sharpwaves.SharpwaveAnalyzer(self.s["sharpwave_analysis_settings"],
                                                            self.fs)
        self.new_dat_index = int(self.fs / self.s["sampling_rate_features"])

    def estimate_features(self, data) -> dict:
        """
        
        Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave
        characteristics.
        
        data (np array) : (channels, time)
        
        returns: 
        dat (pd Dataframe) with naming convention:
            channel_method_feature_(f_band)
        """
        # this is done in a lot of loops unfortunately, 
        # what could be done is to extract the outer channel loop,
        # which could run in parallel
        
        #manager = Manager()
        #features_ = manager.dict() #features_ = {}
        features_ = dict()
        
        # notch filter data before feature estimation 
        if self.s["methods"]["notch_filter"]:
            data = notch_filter(x=data, Fs=self.fs, trans_bandwidth=7,
                                freqs=arange(self.line_noise, 4*self.line_noise,
                                             self.line_noise),
                                fir_design='firwin', verbose=False,
                                notch_widths=3, filter_length=data.shape[1]-1)
        
        if self.s["methods"]["bandpass_filter"]:
            dat_filtered = filter.apply_filter(data, self.filter_fun)  # shape (bands, time)
        else:
            dat_filtered = None

        # mutliprocessing approach
        '''
        job = [Process(target=self.est_ch, args=(features_, ch_idx, ch)) for ch_idx, ch in enumerate(self.ch_names)]
        _ = [p.start() for p in job]
        _ = [p.join() for p in job]
        '''

        #sequential approach
        #for ch_idx, ch in enumerate(self.ch_names):
        for ch_idx in range(len(self.ch_names)):
            ch = self.ch_names[ch_idx]
            features_ = self.est_ch(features_, ch_idx, ch, dat_filtered, data)

        #return dict(features_) # this is necessary for multiprocessing approach 
        return features_
                    
    def est_ch(self, features_, ch_idx, ch, dat_filtered, data) -> dict:
        """Estimate features for a given channel

        Parameters
        ----------
        features_ dict : dict 
            features.py output feature dict
        ch_idx : int
            channel index
        ch : str
            channel name
        dat_filtered : np.ndarray
            notch filtered and bandbass filtered data
        data : np.ndarray)
            notch filtered data

        Returns
        -------
        features_ : dict
            features.py output feature dict
        """

        if self.s["methods"]["bandpass_filter"]:
            features_ = bandpower.get_bandpower_features(
                features_, self.s, self.seglengths, dat_filtered, self.KF_dict,
                ch, ch_idx)

        if self.s["methods"]["raw_hjorth"]: 
            features_ = hjorth_raw.get_hjorth_raw(features_, data[ch_idx, :],
                                                  ch)

        if self.s["methods"]["return_raw"]:
            features_['_'.join([ch, 'raw'])] = data[ch_idx, -1]  # subsampling

        if self.s["methods"]["sharpwave_analysis"]: 
            # print('time taken for sharpwave estimation')
            # start = time.process_time()
            # take only last resampling_rate

            features_ = self.sw_features.get_sharpwave_features(features_,
                data[ch_idx, -self.new_dat_index:], ch)

            # print(time.process_time() - start)
        return features_
