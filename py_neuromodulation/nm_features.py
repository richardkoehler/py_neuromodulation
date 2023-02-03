from __future__ import annotations

"""Module for calculating features."""
import numpy as np

from py_neuromodulation import (
    nm_hjorth_raw,
    nm_sharpwaves,
    nm_coherence,
    nm_fooof,
    nm_nolds,
    nm_features_abc,
    nm_oscillatory,
    nm_bursts,
    nm_linelength,
    nm_mne_connectiviy,
    nm_std,
)


class Features:

    features: list[nm_features_abc.Feature] = []

    def __init__(
        self, s: dict, ch_names: list[str], sfreq: int | float
    ) -> None:
        """Class for calculating features."""
        self.features = []

        for feature in s["features"]:
            if s["features"][feature] is False:
                continue
            if feature == "raw_hjorth":
                self.features.append(nm_hjorth_raw.Hjorth(s, ch_names, sfreq))
            elif feature == "return_raw":
                self.features.append(nm_hjorth_raw.Raw(s, ch_names, sfreq))
            elif feature == "bandpass_filter":
                self.features.append(
                    nm_oscillatory.BandPower(s, ch_names, sfreq)
                )
            elif feature == "stft":
                self.features.append(nm_oscillatory.STFT(s, ch_names, sfreq))
            elif feature == "fft":
                self.features.append(nm_oscillatory.FFT(s, ch_names, sfreq))
            elif feature == "sharpwave_analysis":
                self.features.append(
                    nm_sharpwaves.SharpwaveAnalyzer(s, ch_names, sfreq)
                )
            elif feature == "fooof":
                self.features.append(
                    nm_fooof.FooofAnalyzer(s, ch_names, sfreq)
                )
            elif feature == "nolds":
                self.features.append(nm_nolds.Nolds(s, ch_names, sfreq))
            elif feature == "coherence":
                self.features.append(
                    nm_coherence.NM_Coherence(s, ch_names, sfreq)
                )
            elif feature == "bursts":
                self.features.append(nm_bursts.Burst(s, ch_names, sfreq))
            elif feature == "linelength":
                self.features.append(
                    nm_linelength.LineLength(s, ch_names, sfreq)
                )
            elif feature == "mne_connectivity":
                self.features.append(
                    nm_mne_connectiviy.MNEConnectivity(s, ch_names, sfreq)
                )
            elif feature == "std":
                self.features.append(nm_std.Std(s, ch_names, sfreq))
            else:
                raise ValueError(f"Unknown feature found. Got: {feature}.")

    def estimate_features(self, data: np.ndarray) -> dict:
        """Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave
        characteristics.

        Parameters
        ----------
        data (np array) : (channels, time)

        Returns
        -------
        dat (dict) with naming convention:
            channel_method_feature_(f_band)
        """

        features_compute = {}

        for feature in self.features:
            features_compute = feature.calc_feature(
                data,
                features_compute,
            )

        return features_compute
