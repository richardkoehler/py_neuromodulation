"""Module that contains PNStream ABC."""
import os
import pathlib
from abc import ABC, abstractmethod

import pickle
import pandas as pd
from sklearn import base

from .features import Features
from .processor import DataProcessor
from . import io

_PathLike = str | os.PathLike


class PNStream(ABC):
    settings: dict
    nm_channels: pd.DataFrame
    run_analysis: DataProcessor
    features: Features
    coords: dict
    sfreq: int | float
    path_grids: _PathLike | None
    model: base.BaseEstimator | None
    sess_right: bool | None
    verbose: bool
    PATH_OUT: _PathLike | None
    PATH_OUT_folder_name: _PathLike | None

    def __init__(
        self,
        sfreq: int | float,
        nm_channels: pd.DataFrame | _PathLike,
        settings: dict | _PathLike | None = None,
        line_noise: int | float | None = 50,
        path_grids: _PathLike | None = None,
        coords: dict | None = None,
        coord_names: list | None = None,
        coord_list: list | None = None,
        verbose: bool = True,
    ) -> None:
        self.settings = self._load_settings(settings)
        self.nm_channels = self._load_nm_channels(nm_channels)
        if path_grids is None:
            path_grids = pathlib.Path(__file__).parent.resolve()
        self.path_grids = path_grids
        self.verbose = verbose
        if coords is None:
            self.coords = {}
        else:
            self.coords = coords
        self.sfreq = sfreq
        self.sess_right = None
        self.projection = None
        self.model = None
        self.run_analysis = DataProcessor(
            sfreq=self.sfreq,
            settings=self.settings,
            nm_channels=self.nm_channels,
            path_grids=self.path_grids,
            line_noise=line_noise,
            verbose=self.verbose,
        )

    @abstractmethod
    def run(self):
        """In this function data is first acquired iteratively
        1. self.get_data()
        2. data processing is called:
        self.run_analysis.process_data(data) to calculate features
        3. optional postprocessing
        e.g. plotting, ML estimation is done
        """

    @abstractmethod
    def _add_timestamp(
        self, feature_series: pd.Series, idx: int | None = None
    ) -> pd.Series:
        """Add to feature_series "time" keyword
        For Bids specify with fs_features, for real time analysis with current time stamp
        """

    @staticmethod
    def _get_sess_lat(coords: dict) -> bool:
        if len(coords["cortex_left"]["positions"]) == 0:
            return True
        if len(coords["cortex_right"]["positions"]) == 0:
            return False
        raise ValueError(
            "Either cortex_left or cortex_right positions must be provided."
        )

    @staticmethod
    def _load_nm_channels(
        nm_channels: pd.DataFrame | _PathLike,
    ) -> pd.DataFrame:
        if not isinstance(nm_channels, pd.DataFrame):
            return io.load_nm_channels(nm_channels)
        return nm_channels

    @staticmethod
    def _load_settings(settings: dict | _PathLike | None) -> dict:
        if isinstance(settings, dict):
            return settings
        if settings is None:
            return settings.get_default_settings()
        return io.read_settings(str(settings))

    def load_model(self, model_name: _PathLike) -> None:
        """Load sklearn model, that utilizes predict"""
        with open(model_name, "rb") as fid:
            self.model = pickle.load(fid)

    def save_after_stream(
        self,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
        feature_arr: pd.DataFrame | None = None,
    ) -> None:
        """Save features, settings, nm_channels and sidecar after run"""

        if out_path_root is None:
            out_path_root = os.getcwd()
        # create derivate folder_name output folder if doesn't exist
        if os.path.exists(os.path.join(out_path_root, folder_name)) is False:
            os.makedirs(os.path.join(out_path_root, folder_name))

        self.PATH_OUT = out_path_root
        self.PATH_OUT_folder_name = folder_name
        self.save_sidecar(out_path_root, folder_name)

        if feature_arr is not None:
            self.save_features(out_path_root, folder_name, feature_arr)

        self.save_settings(out_path_root, folder_name)

        self.save_nm_channels(out_path_root, folder_name)

    def save_features(
        self,
        out_path_root: _PathLike,
        folder_name: str,
        feature_arr: pd.DataFrame,
    ) -> None:
        io.save_features(feature_arr, out_path_root, folder_name)

    def save_nm_channels(
        self, out_path_root: _PathLike, folder_name: str
    ) -> None:
        self.run_analysis.save_nm_channels(out_path_root, folder_name)

    def save_settings(
        self, out_path_root: _PathLike, folder_name: str
    ) -> None:
        self.run_analysis.save_settings(out_path_root, folder_name)

    def save_sidecar(self, out_path_root: _PathLike, folder_name: str) -> None:
        """Save sidecar incuding fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'"""
        additional_args = {"sess_right": self.sess_right}
        self.run_analysis.save_sidecar(
            out_path_root, folder_name, additional_args
        )
