import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from py_neuromodulation import nm_IO

target_filter_str = {
    "CLEAN",
    "SQUARED_EMG",
    "SQUARED_INTERPOLATED_EMG",
    "SQUARED_ROTAWHEEL",
    "SQUARED_ROTATION" "rota_squared",
}
features_reverse_order_plotting = {"stft", "fft", "bandpass"}


class Feature_Reader:

    feature_dir: str
    feature_list: list[str]
    settings: dict
    sidecar: dict
    sfreq: int
    line_noise: int
    nm_channels: pd.DataFrame
    feature_arr: pd.DataFrame
    ch_names: list[str]
    ch_names_ECOG: list[str]

    def __init__(
        self, feature_dir: str, feature_file: str, binarize_label: bool = True
    ) -> None:
        """Feature_Reader enables analysis methods on top of NM_reader and NM_Decoder

        Parameters
        ----------
        feature_dir : str, optional
            Path to py_neuromodulation estimated feature runs, where each feature is a folder,
        feature_file : str, optional
            specific feature run, if None it is set to the first feature folder in feature_dir
        """
        self.feature_dir = feature_dir
        self.feature_list = nm_IO.get_run_list_indir(self.feature_dir)
        if feature_file is None:
            self.feature_file = self.feature_list[0]
        else:
            self.feature_file = feature_file

        FILE_BASENAME = Path(self.feature_file).stem
        PATH_READ_FILE = str(
            Path(self.feature_dir, FILE_BASENAME, FILE_BASENAME)
        )

        self.settings = nm_IO.read_settings(PATH_READ_FILE)
        self.sidecar = nm_IO.read_sidecar(PATH_READ_FILE)
        if self.sidecar["sess_right"] is None:
            if "coords" in self.sidecar:
                if len(self.sidecar["coords"]["cortex_left"]["ch_names"]) > 0:
                    self.sidecar["sess_right"] = False
                if len(self.sidecar["coords"]["cortex_right"]["ch_names"]) > 0:
                    self.sidecar["sess_right"] = True
        self.sfreq = self.sidecar["sfreq"]
        self.nm_channels = nm_IO.read_nm_channels(PATH_READ_FILE)
        self.feature_arr = nm_IO.read_features(PATH_READ_FILE)

        self.ch_names = self.nm_channels.new_name
        self.used_chs = list(
            self.nm_channels[
                (self.nm_channels["target"] == 0)
                & (self.nm_channels["used"] == 1)
            ]["new_name"]
        )
        self.ch_names_ECOG = self.nm_channels.query(
            '(type=="ecog") and (used == 1) and (status=="good")'
        ).new_name.to_list()

        if self.nm_channels["target"].sum() > 0:
            self.label_name = self._get_target_ch()
            self.label = self.read_target_ch(
                self.feature_arr,
                self.label_name,
                binarize=binarize_label,
                binarize_th=0.3,
            )

    def _get_target_ch(self) -> str:
        target_names = list(
            self.nm_channels[self.nm_channels["target"] == 1]["name"]
        )
        target_clean = [
            target_name
            for target_name in target_names
            for filter_str in target_filter_str
            if filter_str.lower() in target_name.lower()
        ]

        if len(target_clean) == 0:
            if "ARTIFACT" not in target_names[0]:
                target = target_names[0]
            elif len(target_names) > 1:
                target = target_names[1]
            else:
                target = target_names[0]
        else:
            for target_ in target_clean:
                # try to select contralateral label
                if self.sidecar["sess_right"] is True and "LEFT" in target_:
                    target = target_
                    continue
                elif self.sidecar["sess_right"] is False and "RIGHT" in target_:
                    target = target_
                    continue
                if target_ == target_clean[-1]:
                    target = target_clean[0]  # set label to last element
        return target

    @staticmethod
    def read_target_ch(
        feature_arr: pd.DataFrame,
        label_name: str,
        binarize: bool = True,
        binarize_th: float = 0.3,
    ) -> None:

        label = np.nan_to_num(np.array(feature_arr[label_name]))
        if binarize:
            label = label > binarize_th
        return label

    @staticmethod
    def filter_features(
        feature_columns: list,
        ch_name: str = None,
        list_feature_keywords: list[str] = None,
    ) -> list:
        """filters read features by ch_name and/or modality

        Parameters
        ----------
        feature_columns : list
            [description]
        ch_name : str, optional
            [description], by default None
        list_feature_keywords : list[str], optional
            list of feature strings that need to be in the columns, by default None

        Returns
        -------
        list
            column list that suffice the ch_name and list_feature_keywords
        """

        if ch_name is not None:
            feature_select = [i for i in list(feature_columns) if ch_name in i]
        else:
            feature_select = feature_columns

        if list_feature_keywords is not None:
            feature_select = [
                f
                for f in feature_select
                if any(x in f for x in list_feature_keywords)
            ]

            if (
                len(
                    [
                        mod
                        for mod in features_reverse_order_plotting
                        if mod in list_feature_keywords
                    ]
                )
                > 0
            ):
                # flip list s.t. theta band is lowest in subsequent plot
                feature_select = feature_select[::-1]

        return feature_select

    def set_target_ch(self, ch_name: str) -> None:
        self.label = ch_name

    def normalize_features(
        self,
    ) -> pd.DataFrame:
        """Normalize feature_arr feature columns

        Returns:
            pd.DataFrame: z-scored feature_arr
        """
        cols_norm = [c for c in self.feature_arr.columns if "time" not in c]
        feature_arr_norm = stats.zscore(self.feature_arr[cols_norm])
        feature_arr_norm["time"] = self.feature_arr["time"]
        return feature_arr_norm

    @staticmethod
    def get_performace_sub_strip(performance_sub: dict, plt_grid: bool = False):

        ecog_strip_performance = []
        ecog_coords_strip = []
        cortex_grid = []
        grid_performance = []

        channels_ = performance_sub.keys()

        for ch in channels_:
            if "grid" not in ch and "combined" not in ch:
                ecog_coords_strip.append(performance_sub[ch]["coord"])
                ecog_strip_performance.append(
                    performance_sub[ch]["performance_test"]
                )
            elif plt_grid is True and "gridcortex_" in ch:
                cortex_grid.append(performance_sub[ch]["coord"])
                grid_performance.append(performance_sub[ch]["performance_test"])

        if len(ecog_coords_strip) > 0:
            ecog_coords_strip = np.vstack(ecog_coords_strip)

        return (
            ecog_strip_performance,
            ecog_coords_strip,
            cortex_grid,
            grid_performance,
        )

    @staticmethod
    def get_epochs(data, y_, epoch_len, sfreq, threshold=0):
        """Return epoched data.

        Parameters
        ----------
        data : np.ndarray
            array of extracted features of shape (n_samples, n_channels, n_features)
        y_ : np.ndarray
            array of labels e.g. ones for movement and zeros for
            no movement or baseline corr. rotameter data
        sfreq : int/float
            sampling frequency of data
        epoch_len : int
            length of epoch in seconds
        threshold : int/float
            (Optional) threshold to be used for identifying events
            (default=0 for y_tr with only ones
            and zeros)

        Returns
        -------
        epoch_ np.ndarray
            array of epoched ieeg data with shape (epochs,samples,channels,features)
        y_arr np.ndarray
            array of epoched event label data with shape (epochs,samples)
        """

        epoch_lim = int(epoch_len * sfreq)

        ind_mov = np.where(np.diff(np.array(y_ > threshold) * 1) == 1)[0]

        low_limit = ind_mov > epoch_lim / 2
        up_limit = ind_mov < y_.shape[0] - epoch_lim / 2

        ind_mov = ind_mov[low_limit & up_limit]

        epoch_ = np.zeros(
            [ind_mov.shape[0], epoch_lim, data.shape[1], data.shape[2]]
        )

        y_arr = np.zeros([ind_mov.shape[0], int(epoch_lim)])

        for idx, i in enumerate(ind_mov):

            epoch_[idx, :, :, :] = data[
                i - epoch_lim // 2 : i + epoch_lim // 2, :, :
            ]

            y_arr[idx, :] = y_[i - epoch_lim // 2 : i + epoch_lim // 2]

        return epoch_, y_arr

    def read_results(
        self,
        performance_dict: dict = {},
        subject_name: str = None,
        DEFAULT_PERFORMANCE: float = 0.5,
        read_grid_points: bool = True,
        read_channels: bool = True,
        read_all_combined: bool = False,
        ML_model_name: str = "LM",
        read_mov_detection_rates: bool = False,
        read_bay_opt_params: bool = False,
        read_mrmr: bool = False,
        model_save: bool = False,
        save_results: bool = False,
        PATH_OUT: str = None,
        folder_name: str = None,
        str_add: str = None,
    ):
        """Save performances of a given patient into performance_dict from saved nm_decoder

        Parameters
        ----------
        performance_dict : dictionary
            dictionary including decoding performances, by default dictionary
        subject_name : string, optional
            subject name, by default None
        feature_file : string, optional
            feature file, by default None
        DEFAULT_PERFORMANCE : float, optional
            chance performance, by default 0.5
        read_grid_points : bool, optional
            true if grid point performances are read, by default True
        read_channels : bool, optional
            true if channels performances are read, by default True
        read_all_combined : bool, optional
            true if all combined channel performances are read, by default False
        ML_model_name : str, optional
            machine learning model name, by default 'LM'
        read_mov_detection_rates : boolean, by defaulte False
            if True, read movement detection rates, as well as fpr's and tpr's
        Returns
        -------
        dictionary
            performance_dict
        """

        if ".vhdr" in self.feature_file:
            feature_file = self.feature_file[: -len(".vhdr")]
        else:
            feature_file = self.feature_file

        if subject_name is None:
            subject_name = feature_file[
                feature_file.find("sub-") : feature_file.find("_ses")
            ][4:]

        PATH_ML_ = os.path.join(
            self.feature_dir,
            feature_file,
            feature_file + "_" + ML_model_name + "_ML_RES.p",
        )

        performance_dict[subject_name] = {}

        def write_CV_res_in_performance_dict(
            obj_read,
            obj_write,
            read_mov_detection_rates=read_mov_detection_rates,
            read_bay_opt_params=False,
        ):
            def transform_list_of_dicts_into_dict_of_lists(l_):
                dict_out = {}
                for key_, _ in l_[0].items():
                    key_l = []
                    for dict_ in l_:
                        key_l.append(dict_[key_])
                    dict_out[key_] = key_l
                return dict_out

            def read_ML_performances(
                obj_read, obj_write, set_inner_CV_res: bool = False
            ):
                def set_score(
                    key_set: str,
                    key_get: str,
                    take_mean: bool = True,
                    val=None,
                ):
                    if set_inner_CV_res is True:
                        key_set = "InnerCV_" + key_set
                        key_get = "InnerCV_" + key_get
                    if take_mean is True:
                        val = np.mean(obj_read[key_get])
                    obj_write[key_set] = val

                set_score(
                    key_set="performance_test",
                    key_get="score_test",
                    take_mean=True,
                )
                set_score(
                    key_set="performance_train",
                    key_get="score_train",
                    take_mean=True,
                )

                if "coef" in obj_read:
                    set_score(
                        key_set="coef",
                        key_get="coef",
                        take_mean=False,
                        val=np.concatenate(obj_read["coef"]),
                    )

                if read_mov_detection_rates:
                    set_score(
                        key_set="mov_detection_rates_test",
                        key_get="mov_detection_rates_test",
                        take_mean=True,
                    )
                    set_score(
                        key_set="mov_detection_rates_train",
                        key_get="mov_detection_rates_train",
                        take_mean=True,
                    )
                    set_score(
                        key_set="fprate_test",
                        key_get="fprate_test",
                        take_mean=True,
                    )
                    set_score(
                        key_set="fprate_train",
                        key_get="fprate_train",
                        take_mean=True,
                    )
                    set_score(
                        key_set="tprate_test",
                        key_get="tprate_test",
                        take_mean=True,
                    )
                    set_score(
                        key_set="tprate_train",
                        key_get="tprate_train",
                        take_mean=True,
                    )

                if read_bay_opt_params is True:
                    # transform dict into keys for json saving
                    dict_to_save = transform_list_of_dicts_into_dict_of_lists(
                        obj_read["best_bay_opt_params"]
                    )
                    set_score(
                        key_set="bay_opt_best_params",
                        key_get=None,
                        take_mean=False,
                        val=dict_to_save,
                    )

                if read_mrmr is True:
                    # transform dict into keys for json saving

                    set_score(
                        key_set="mrmr_select",
                        key_get=None,
                        take_mean=False,
                        val=obj_read["mrmr_select"],
                    )
                if model_save is True:
                    set_score(
                        key_set="model_save",
                        key_get=None,
                        take_mean=False,
                        val=obj_read["model_save"],
                    )

            read_ML_performances(obj_read, obj_write)

            if (
                len([key_ for key_ in obj_read.keys() if "InnerCV_" in key_])
                > 0
            ):
                read_ML_performances(obj_read, obj_write, set_inner_CV_res=True)

        if read_channels:

            ch_to_use = self.ch_names_ECOG
            for ch in ch_to_use:

                performance_dict[subject_name][ch] = {}

                if "coords" in self.sidecar:
                    if (
                        len(self.sidecar["coords"]) > 0
                    ):  # check if coords are empty

                        coords_exist = False
                        for cortex_loc in self.sidecar["coords"].keys():
                            for ch_name_coord_idx, ch_name_coord in enumerate(
                                self.sidecar["coords"][cortex_loc]["ch_names"]
                            ):
                                if ch.startswith(ch_name_coord):
                                    coords = self.sidecar["coords"][cortex_loc][
                                        "positions"
                                    ][ch_name_coord_idx]
                                    coords_exist = True  # optimally break out of the two loops...
                        if coords_exist is False:
                            coords = None
                        performance_dict[subject_name][ch]["coord"] = coords
                write_CV_res_in_performance_dict(
                    ML_res.ch_ind_results[ch],
                    performance_dict[subject_name][ch],
                    read_mov_detection_rates=read_mov_detection_rates,
                    read_bay_opt_params=read_bay_opt_params,
                )

        if read_all_combined:
            performance_dict[subject_name]["all_ch_combined"] = {}
            write_CV_res_in_performance_dict(
                ML_res.all_ch_results,
                performance_dict[subject_name]["all_ch_combined"],
                read_mov_detection_rates=read_mov_detection_rates,
                read_bay_opt_params=read_bay_opt_params,
            )

        if read_grid_points:
            performance_dict[subject_name][
                "active_gridpoints"
            ] = ML_res.active_gridpoints

            for project_settings, grid_type in zip(
                ["project_cortex", "project_subcortex"],
                ["gridcortex_", "gridsubcortex_"],
            ):
                if self.settings["postprocessing"][project_settings] is False:
                    continue

                # the sidecar keys are grid_cortex and subcortex_grid
                for grid_point in range(
                    len(self.sidecar["grid_" + project_settings.split("_")[1]])
                ):

                    gp_str = grid_type + str(grid_point)

                    performance_dict[subject_name][gp_str] = {}
                    performance_dict[subject_name][gp_str][
                        "coord"
                    ] = self.sidecar["grid_" + project_settings.split("_")[1]][
                        grid_point
                    ]

                    if gp_str in ML_res.active_gridpoints:
                        write_CV_res_in_performance_dict(
                            ML_res.gridpoint_ind_results[gp_str],
                            performance_dict[subject_name][gp_str],
                            read_mov_detection_rates=read_mov_detection_rates,
                            read_bay_opt_params=read_bay_opt_params,
                        )
                    else:
                        # set non interpolated grid point to default performance
                        performance_dict[subject_name][gp_str][
                            "performance_test"
                        ] = DEFAULT_PERFORMANCE
                        performance_dict[subject_name][gp_str][
                            "performance_train"
                        ] = DEFAULT_PERFORMANCE

        if save_results:
            nm_IO.save_general_dict(
                dict_=performance_dict,
                path_out=PATH_OUT,
                str_add=str_add,
                folder_name=folder_name,
            )
        return performance_dict

    @staticmethod
    def get_dataframe_performances(p: dict) -> pd.DataFrame:
        performances = []
        for sub in p.keys():
            for ch in p[sub].keys():
                if "active_gridpoints" in ch:
                    continue
                dict_add = p[sub][ch].copy()
                dict_add["sub"] = sub
                dict_add["ch"] = ch

                if "all_ch_" in ch:
                    dict_add["ch_type"] = "all ch combinded"
                elif "gridcortex" in ch:
                    dict_add["ch_type"] = "cortex grid"
                else:
                    dict_add["ch_type"] = "electrode ch"
                performances.append(dict_add)
        df = pd.DataFrame(performances)

        return df
