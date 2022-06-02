import numpy as np
from typing import Iterable

from py_neuromodulation import nm_features_abc, nm_oscillatory


class Nolds(nm_features_abc.Feature):

    def __init__(self, settings: dict, ch_names: Iterable[str], sfreq: float) -> None:
        self.s = settings
        self.ch_names = ch_names

        if len(self.s["nolds_features"]["data"]["frequency_bands"]) > 0:
            self.bp_filter = nm_oscillatory.BandPower(settings, ch_names, sfreq, use_kf=False)        

    def calc_feature(self, data: np.array, features_compute: dict,
    ) -> dict:

        if self.s["nolds_features"]["data"]["raw"]:
            features_compute = self.calc_nolds(data, features_compute)
        if len(self.s["nolds_features"]["data"]["frequency_bands"]) > 0:
            data_filt = self.bp_filter.bandpass_filter.filter_data(data)

            for f_band_idx, f_band in enumerate(self.s["nolds_features"]["data"]["frequency_bands"]):
                # filter data now for a specific fband and pass to calc_nolds
                features_compute = self.calc_nolds(data_filt[:, f_band_idx, :], features_compute, f_band)  # ch, bands, samples
        return features_compute

    def calc_nolds(self, data: np.array, features_compute: dict, data_str: str = "raw"
    ) -> dict:

        for ch_idx, ch_name in enumerate(self.ch_names):
            if self.s["nolds_features"]["sample_entropy"]:
                features_compute[f"nolds_{ch_name}_sample_entropy"] = nolds.sampen(
                    data[ch_idx, :]
                )
            if self.s["nolds_features"]["correlation_dimension"]:
                features_compute[
                    f"nolds_{ch_name}_correlation_dimension_{data_str}"
                ] = nolds.corr_dim(data[ch_idx, :], emb_dim=2)
            if self.s["nolds_features"]["lyapunov_exponent"]:
                features_compute[f"nolds_{ch_name}_lyapunov_exponent_{data_str}"] = nolds.lyap_r(
                    data[ch_idx, :]
                )
            if self.s["nolds_features"]["hurst_exponent"]:
                features_compute[f"nolds_{ch_name}_hurst_exponent_{data_str}"] = nolds.hurst_rs(
                    data[ch_idx, :]
                )
            if self.s["nolds_features"]["detrended_fluctutaion_analysis"]:
                features_compute[
                    f"nolds_{ch_name}_detrended_fluctutaion_analysis_{data_str}"
                ] = nolds.dfa(data[ch_idx, :])

        return features_compute
