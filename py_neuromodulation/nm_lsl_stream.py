from __future__ import annotations
import multiprocessing
import pandas as pd
import pylsl
import numpy as np

from py_neuromodulation import nm_stream_abc


class StreamNotFoundException(Exception):
    def __str__(self) -> str:
        return "No Stream found"


# Note: For multiprocessing pylsl can only be used within the main thread
# There instantiation of the lsl_inlet would need to be pulled out
# into the main app starting streaming
# https://github.com/chkothe/pylsl/issues/7


class LSLStream(nm_stream_abc.PNStream):
    def init_LSL(
        self,
        buffer_size: int = 250,
        wait_time: int = 2,
        max_samples: int = 250,
        time_out: int = 1,
    ):

        self.wait_time = wait_time
        self.buffer_size = buffer_size
        self.time_out = time_out
        self.max_samples = max_samples

        streams = pylsl.resolve_streams(
            wait_time=wait_time,
        )
        if len(streams) > 0:
            print("Stream found")
        else:
            print("No Stream found")
            raise StreamNotFoundException

        self.lsl_streaminlet = pylsl.StreamInlet(
            info=streams[0], max_buflen=buffer_size
        )

    def run(self, raw_data: np.array):
        feature_series = self.run_analysis.process(raw_data)
        return feature_series

    def get_data(
        self,
    ):
        """Data is pulled by lsl, and is returned in shape (ch, time)

        Parameters
        ----------
        queue_raw : multiprocessing.Queue
            _description_
        """
        samples, _ = self.lsl_streaminlet.pull_chunk(
            max_samples=self.max_samples, timeout=self.time_out
        )
        raw_data = np.array(samples).T
        return raw_data

    def disconnect_lsl(self):
        self.lsl_streaminlet.close_stream()

    def _add_timestamp(
        self, feature_series: pd.Series, idx: int | None = None
    ) -> pd.Series:
        ...
