import math
import os
import numpy as np
from numpy.testing import assert_allclose

from py_neuromodulation.nm_filter import IIRFilter

from py_neuromodulation import (
    nm_generator,
    nm_settings,
    nm_IO,
    nm_define_nmchannels,
)

from matplotlib import pyplot as plt
from matplotlib import mlab


class TestIIRFilter:
    def setUp(self) -> None:
        """This test function sets a data batch and automatic initialized M1 datafram

        Args:
            PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

        Returns:
            ieeg_batch (np.ndarray): (channels, samples)
            df_M1 (pd Dataframe): auto intialized table for rereferencing
            settings_wrapper (settings.py): settings.json
            fs (float): example sampling frequency
        """
        sub = "000"
        ses = "right"
        task = "force"
        run = 3
        datatype = "ieeg"

        # Define run name and access paths in the BIDS format.
        RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{datatype}.vhdr"

        PATH_RUN = os.path.join(
            os.path.abspath(os.path.join("examples", "data")),
            f"sub-{sub}",
            f"ses-{ses}",
            datatype,
            RUN_NAME,
        )
        PATH_BIDS = os.path.abspath(os.path.join("examples", "data"))

        (
            raw,
            data,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
        ) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
        )
        self.sfreq = math.floor(sfreq)
        self.nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference="default",
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog", "dbs", "seeg"),
            target_keywords=("SQUARED_ROTATION",),
        )

        settings = nm_settings.get_default_settings()
        # settings = nm_settings.set_settings_fast_compute(
        #     settings
        # )  # includes rereference features

        self.generator = nm_generator.raw_data_generator(
            data, settings, self.sfreq
        )
        self.data_batch = next(self.generator, None)

    def test_notch(self) -> None:
        """
        Args:
            ref_here (RT_rereference): Rereference initialized object
            ieeg_batch (np.ndarray): sample data
            df_M1 (pd.Dataframe): rereferencing dataframe
        """
        self.setUp()
        preprocessor = IIRFilter(
            sfreq=self.sfreq,
            order=2,
            l_freq=48,
            h_freq=52,
            filter_type="bandstop",
        )
        data = self.data_batch
        data_filt = preprocessor.process(data)
        data = data[5, :]
        data_filt = data_filt[5, :]

        plt.plot(data)
        plt.show(block=True)
        plt.plot(data_filt)
        plt.show(block=True)

        s, f = mlab.psd(data, NFFT=self.sfreq // 2, Fs=self.sfreq)
        plt.loglog(f, s)
        plt.axvline(50)
        plt.grid(True)
        plt.show(block=True)
        s, f = mlab.psd(data_filt, NFFT=self.sfreq // 2, Fs=self.sfreq)
        plt.loglog(f, s)
        plt.axvline(50)
        plt.grid(True)
        plt.show(block=True)
        ...
        dat_filt_2 = preprocessor.process(next(self.generator, None))

        # assert_allclose(dat_filt, ref_dat_old, rtol=1e-7, equal_nan=False)


if __name__ == "__main__":
    test = TestIIRFilter()
    test.test_notch()
