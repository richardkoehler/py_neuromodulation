import os
import sys
import py_neuromodulation as nm
import xgboost
import pickle
import numpy as np
from pprint import pprint

from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_stream_offline,
    nm_IO,
    nm_plots,
    nm_filter,
)
from sklearn import metrics, model_selection
from matplotlib import pyplot as plt

if __name__ == "__main__":

    with open("/Users/hi/Downloads/PIT-RNS1438.p", "rb") as handle:
        d = pickle.load(handle)

    print()
    PEs = list(d.keys())
    PE = PEs[0]
    recs = list(d[PE].keys())
    rec = recs[0]
    dat = d[PE][rec]

    dat_t = dat["dat_epoch"]
    # plt.plot(dat_t[0, :])

    ch_names = ["ch1", "ch2", "ch3", "ch4"]
    ch_types = ["ecog" for _ in range(len(ch_names))]
    sfreq = 250

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference=None,
        bads=None,
        new_names="default",
        used_types=["ecog"],
    )

    stream = nm_stream_offline.Stream(
        settings=None,
        nm_channels=nm_channels,
        verbose=True,  # Change here if you want to see the outputs of the run
    )

    stream.set_settings_fast_compute()

    stream.settings["preprocessing"]["re_referencing"] = False
    stream.settings["postprocessing"]["feature_normalization"] = False
    stream.settings["preprocessing"]["raw_resampling"] = False
    stream.settings["preprocessing"]["notch_filter"] = True
    stream.settings["preprocessing"]["preprocessing_order"] = ["notch_filter"]

    stream.settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "alpha": [8, 12],
        "low beta": [13, 20],
        "high beta": [20, 35],
        "low gamma": [35, 55],
    }

    notch_filter_init = nm_filter.NotchFilter(
        sfreq=sfreq,
        line_noise=60,
        notch_widths=5,
        trans_bandwidth=10,
        freqs=np.array(
            [
                60,
            ]
        ),
    )

    pprint(stream.settings)

    stream.init_stream(
        sfreq=sfreq, line_noise=60, notch_filter=notch_filter_init
    )

    data = dat["dat_epoch"]
    PATH_OUT = "/Users/hi/Downloads/PlayAround"

    stream.run(
        data=data,
        folder_name="FirstSub",
        out_path_root=PATH_OUT,
    )

    print("check features")

    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file="FirstSub"
    )

    # We arbitrarily decided to plot the features from the first ECoG channel
    ch_name = 'ch1'
    # In order to make nice labels in the y axis we do:
    feature_names = list(feature_reader.feature_arr.filter(regex=ch_name)[1:].columns)
    feature_col_name = [
        i[len(ch_name) + 1 :] for i in feature_names if ch_name in i
    ]
    plt.figure()  # figsize=(20,15)
    # Here we remove the first data point from all features, because the normalization only starts in the second data point.
    plt.imshow(feature_reader.feature_arr.filter(regex=ch_name)[1:].T, aspect='auto')
    plt.yticks(np.arange(0, len(feature_names), 1), feature_col_name)
    plt.title("Estimated features over time for channel {}".format(ch_name))
    plt.xlabel("Time points")
    plt.tight_layout()