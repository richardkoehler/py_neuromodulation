import os
from re import VERBOSE
from py_neuromodulation import (
    nm_RealTimeClientStreamSingleProcess,
    nm_RealTimeClientStreamMultiprocessing
)

if __name__ == "__main__":

    #nm_RealTimeClientStreamMock
    stream = nm_RealTimeClientStreamSingleProcess.RealTimePyNeuro(
        PATH_SETTINGS=os.path.abspath("examples/rt_example/nm_settings.json"),
        PATH_NM_CHANNELS=os.path.abspath("examples/rt_example/nm_channels.csv"),
        PATH_OUT=os.path.abspath("examples/rt_example"),
        VERBOSE=False
    )

    stream.run()
