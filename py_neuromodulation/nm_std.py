from __future__ import annotations
from typing import Iterable, Sequence

import numpy as np
from py_neuromodulation import nm_features_abc


class Std(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float, windows: Sequence | None = None
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        if windows is None:
            windows = (settings["segment_length_features_ms"],)

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for window in windows:
                samples = -int(np.floor(window / 1000 * sfreq))
                feature_name = "_".join([ch_name, "std", f"{window}ms"])
                self.feature_params.append((ch_idx, samples, feature_name))

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        for ch_idx, samples, feature_name in self.feature_params:
            features_compute[feature_name] = np.std(data[ch_idx, samples:])
        return features_compute
