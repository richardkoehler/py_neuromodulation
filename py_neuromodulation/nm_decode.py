from numpy.core.overrides import verify_matching_signatures
from scipy.sparse import csc
from sklearn import model_selection, metrics, linear_model, discriminant_analysis, base
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, Optimizer
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn.utils import class_weight
from scipy.ndimage import (binary_dilation,
                           binary_erosion)
from scipy.ndimage import label as label_ndimage
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import os
import json
import numpy as np
import xgboost
from typing import Type
import _pickle as cPickle

class CV_res:
    def __init__(
        self,
        get_movement_detection_rate: bool = False,
        RUN_BAY_OPT : bool = False
    ) -> None:

        self.score_train = []
        self.score_test = []
        self.y_test = []
        self.y_train = []
        self.y_test_pr = []
        self.y_train_pr = []
        self.X_test = []
        self.X_train = []
        self.coef = []
        if get_movement_detection_rate is True:
            self.mov_detection_rates_test = []
            self.tprate_test = []
            self.fprate_test = []
            self.mov_detection_rates_train = []
            self.tprate_train = []
            self.fprate_train = []
        if RUN_BAY_OPT is True:
            self.best_bay_opt_params = []

class Decoder:

    features: pd.DataFrame
    label: np.ndarray
    model: base.BaseEstimator
    cv_method: model_selection.BaseCrossValidator
    use_nested_cv : bool
    threshold_score: bool
    mov_detection_threshold: float
    TRAIN_VAL_SPLIT: bool
    RUN_BAY_OPT: bool
    save_coef: bool
    get_movement_detection_rate: bool
    min_consequent_count: int
    STACK_FEATURES_N_SAMPLES: bool
    time_stack_n_samples: int
    ros: RandomOverSampler = None
    VERBOSE : bool = False
    ch_ind_data : dict = {}
    grid_point_ind_data : dict = {}
    active_gridpoints : list = []
    feature_names : list = []
    ch_ind_results : dict = {}
    gridpoint_ind_results : dict = {}
    all_ch_results : dict = {}

    class ClassMissingException(Exception):
        def __init__(
            self,
            message="Only one class present.",
        ) -> None:
            self.message = message
            super().__init__(self.message)

        def __str__(self):
            print(self.message)

    def __init__(self,
                 features: pd.DataFrame = None,
                 label: np.ndarray = None,
                 label_name: str = None,
                 used_chs: list[str]=None,
                 model=linear_model.LinearRegression(),
                 eval_method=metrics.r2_score,
                 cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                 use_nested_cv : bool = False,
                 threshold_score=True,
                 mov_detection_threshold:float =0.5,
                 TRAIN_VAL_SPLIT: bool=True,
                 RUN_BAY_OPT: bool=False,
                 STACK_FEATURES_N_SAMPLES: bool=True,
                 time_stack_n_samples: int = 5,
                 save_coef:bool =False,
                 get_movement_detection_rate:bool =False,
                 min_consequent_count:int =3,
                 bay_opt_param_space: list = [],
                 VERBOSE: bool = False) -> None:
        """Initialize here a feature file for processing
        Read settings.json nm_channels.csv and features.csv
        Read target label

        Parameters
        ----------
        model : machine learning model
            model that utilizes fit and predict functions
        eval_method : sklearn metrics
            evaluation scoring method
        cv_method : sklearm model_selection method
        threshold_score : boolean
            if True set lower threshold at zero (useful for r2),
        mov_detection_threshold : float
            if get_movement_detection_rate is True, find given minimum 'threshold' respective 
            consecutive movement blocks, by default 0.5
        TRAIN_VAL_SPLIT (boolean):
            if true split data into additinal validation, and run class weighted CV
        save_coef (boolean):
            if true, save model._coef trained coefficients
        get_movement_detection_rate (boolean):
            save detection rate and tpr / fpr as well
        min_consequent_count (int):
            if get_movement_detection_rate is True, find given 'min_consequent_count' respective 
            consecutive movement blocks with minimum size of 'min_consequent_count'
        """

        if any(e is not None for e in [features, label_name]):
            self.set_data(features, label, label_name, used_chs)

        self.model = model
        self.eval_method = eval_method
        self.cv_method = cv_method
        self.use_nested_cv = use_nested_cv
        self.threshold_score = threshold_score
        self.mov_detection_threshold = mov_detection_threshold
        self.TRAIN_VAL_SPLIT = TRAIN_VAL_SPLIT
        self.RUN_BAY_OPT = RUN_BAY_OPT
        self.save_coef = save_coef
        self.get_movement_detection_rate = get_movement_detection_rate
        self.min_consequent_count = min_consequent_count
        self.STACK_FEATURES_N_SAMPLES = STACK_FEATURES_N_SAMPLES
        self.time_stack_n_samples = time_stack_n_samples
        self.bay_opt_param_space = bay_opt_param_space
        self.VERBOSE = VERBOSE

        self.ch_ind_data = {}
        self.grid_point_ind_data = {}
        self.active_gridpoints= []
        self.feature_names = []
        self.ch_ind_results = {}
        self.gridpoint_ind_results = {}
        self.all_ch_results = {}

        if type(self.model) is discriminant_analysis.LinearDiscriminantAnalysis:
            self.ros = RandomOverSampler(random_state=0)

    def set_data(self, features, label, label_name, used_chs):

        self.features = features
        self.label = label
        self.label_name = label_name
        self.data = np.nan_to_num(
                np.array(
                    self.features[
                        [
                            col for col in self.features.columns
                            if not (('time' in col) or (self.label_name in col))
                        ]
                    ]
                )
            )
        self.used_chs = used_chs

    def set_data_ind_channels(self):
        """specified channel individual data
        """
        self.ch_ind_data = {}
        for ch in self.used_chs:
            self.ch_ind_data[ch] = np.nan_to_num(
                np.array(
                    self.features[
                        [col for col in self.features.columns if col.startswith(ch)]
                    ]
                )
            )

    def set_CV_results(self, attr_name, contact_point=None):
        """set CV results in respectie nm_decode attributes
        The reference is first stored in obj_set, and the used lateron

        Parameters
        ----------
        attr_name : string
            is either all_ch_results, ch_ind_results, gridpoint_ind_results
        contact_point : object, optional
            usually an int specifying the grid_point or string, specifying the used channel,
            by default None
        """
        if contact_point is not None:
            getattr(self, attr_name)[contact_point] = {}
            obj_set = getattr(self, attr_name)[contact_point]
        else:
            obj_set = getattr(self, attr_name)

        def set_scores(cv_res: Type[CV_res], set_inner_CV_res : bool = False):

            def set_score(key_ : str, val):
                if set_inner_CV_res is True:
                    key_ = "InnerCV_" + key_
                obj_set[key_] = val

            set_score("score_train", cv_res.score_train)
            set_score("score_test", cv_res.score_test)
            set_score("y_test", cv_res.y_test)
            set_score("y_train", cv_res.y_train)
            set_score("y_test_pr", cv_res.y_test_pr)
            set_score("y_train_pr", cv_res.y_train_pr)
            set_score("X_train", cv_res.X_train)
            set_score("X_test", cv_res.X_test)

            if self.save_coef:
                set_score("coef", cv_res.coef)
            if self.get_movement_detection_rate:
                set_score("mov_detection_rates_test", cv_res.mov_detection_rates_test)
                set_score("mov_detection_rates_train", cv_res.mov_detection_rates_train)
                set_score("fprate_test", cv_res.fprate_test)
                set_score("fprate_train", cv_res.fprate_train)
                set_score("tprate_test", cv_res.tprate_test)
                set_score("tprate_train", cv_res.tprate_train)

            if self.RUN_BAY_OPT is True:
                set_score("best_bay_opt_params", cv_res.best_bay_opt_params)
            return obj_set

        obj_set = set_scores(self.cv_res)

        if self.use_nested_cv is True:
            obj_set = set_scores(self.cv_res_inner, set_inner_CV_res=True)

    def run_CV_caller(self, feature_contacts: str="ind_channels"):
        """Wrapper that call for all channels / grid points / combined channels the CV function

        Parameters
        ----------
        feature_contacts : str, optional
            "grid_points", "ind_channels" or "all_channels_combined" , by default "ind_channels"
        """
        valid_feature_contacts = ["ind_channels", "all_channels_combined", "grid_points"]
        if feature_contacts not in valid_feature_contacts:
            raise ValueError(f"{feature_contacts} not in {valid_feature_contacts}")

        if feature_contacts == "grid_points":
            for grid_point in self.active_gridpoints:
                self.run_CV(self.grid_point_ind_data[grid_point], self.label)
                self.set_CV_results('gridpoint_ind_results', contact_point=grid_point)
            return self.gridpoint_ind_results

        if feature_contacts == "ind_channels":
            for ch in self.used_chs:
                self.run_CV(self.ch_ind_data[ch], self.label)
                self.set_CV_results('ch_ind_results', contact_point=ch)
            return self.ch_ind_results

        if feature_contacts == "all_channels_combined":
            dat_combined = np.concatenate(list(self.ch_ind_data.values()), axis=1)
            self.run_CV(dat_combined, self.label)
            self.set_CV_results('all_ch_results', contact_point=None)
            return self.all_ch_results

    def set_data_grid_points(self, cortex_only=False, subcortex_only=False):
        """Read the run_analysis
        Projected data has the shape (samples, grid points, features)
        """

        # activate_gridpoints stores cortex + subcortex data
        self.active_gridpoints = np.unique(
            [i.split('_')[0] + "_" + i.split('_')[1]
            for i in self.features.columns 
                if "grid" in i]
        )

        if cortex_only:
            self.active_gridpoints = [
                i
                for i in self.active_gridpoints 
                if i.startswith("gridcortex")
            ]

        if subcortex_only:
            self.active_gridpoints = [
                i
                for i in self.active_gridpoints 
                if i.startswith("gridsubcortex")
            ]

        self.feature_names = [
            i[len(self.active_gridpoints[0]+"_"):] 
            for i in self.features.columns 
                if self.active_gridpoints[0]+"_" in i
        ]

        self.grid_point_ind_data = {}

        self.grid_point_ind_data = {
            grid_point : np.nan_to_num(self.features[
                    [i 
                    for i in self.features.columns 
                        if grid_point in i]
                    ]
            )
            for grid_point in self.active_gridpoints
        }

    def get_movement_grouped_array(self, prediction, threshold=0.5, min_consequent_count=5):
        """Return given a 1D numpy array, an array of same size with grouped consective blocks

        Parameters
        ----------
        prediction : np.array
            numpy array of either predictions or labels, that is going to be grouped
        threshold : float, optional
            threshold to be applied to 'prediction', by default 0.5
        min_consequent_count : int, optional
            minimum required consective samples higher than 'threshold', by default 5

        Returns
        -------
        labeled_array : np.array
            grouped vector with incrementing number for movement blocks
        labels_count : int
            count of individual movement blocks
        """
        mask = prediction > threshold
        structure = [True] * min_consequent_count  # used for erosion and dilation
        eroded = binary_erosion(mask, structure)
        dilated = binary_dilation(eroded, structure)
        labeled_array, labels_count = label_ndimage(dilated)
        return labeled_array, labels_count

    def calc_movement_detection_rate(self, y_label, prediction, threshold=0.5, min_consequent_count=3):
        """Given a label and prediction, return the movement detection rate on the basis of 
        movements classified in blocks of 'min_consequent_count'.

        Parameters
        ----------
        y_label : [type]
            [description]
        prediction : [type]
            [description]
        threshold : float, optional
            threshold to be applied to 'prediction', by default 0.5
        min_consequent_count : int, optional
            minimum required consective samples higher than 'threshold', by default 3

        Returns
        -------
        mov_detection_rate : float
            movement detection rate, where at least 'min_consequent_count' samples where high in prediction
        fpr : np.array
            sklearn.metrics false positive rate np.array
        tpr : np.array
            sklearn.metrics true positive rate np.array
        """

        pred_grouped, _ = self.get_movement_grouped_array(
            prediction,
            threshold,
            min_consequent_count
        )
        y_grouped, labels_count = self.get_movement_grouped_array(
            y_label,
            threshold,
            min_consequent_count
        )

        hit_rate = np.zeros(labels_count)
        pred_group_bin = np.array(pred_grouped>0)

        for label_number in range(1, labels_count + 1):  # labeling starts from 1    
            hit_rate[label_number-1] = np.sum(
                pred_group_bin[np.where(y_grouped == label_number)[0]]
            )

        try:
            mov_detection_rate = np.where(hit_rate>0)[0].shape[0] / labels_count
        except ZeroDivisionError:
            print("no movements in label")
            return 0, 0, 0

        # calculating TPR and FPR: https://stackoverflow.com/a/40324184/5060208
        CM = metrics.confusion_matrix(y_label, prediction)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)

        return mov_detection_rate, fpr, tpr

    def init_cv_res(self) -> None:
        return CV_res(
            self.get_movement_detection_rate,
            self.RUN_BAY_OPT
        )

    @staticmethod
    def append_previous_n_samples(X: np.ndarray, y: np.ndarray, n: int = 5):
        """
        stack feature vector for n samples
        """
        time_arr = np.zeros([X.shape[0]-n, int(n*X.shape[1])])
        for time_idx, time_ in enumerate(np.arange(n, X.shape[0])):
            for time_point in range(n):
                time_arr[time_idx, time_point*X.shape[1]:(time_point+1)*X.shape[1]] = \
                    X[time_-time_point,:]
        return time_arr, y[n:]

    @staticmethod
    def append_samples_val(X_train, y_train, X_val, y_val, n):

        X_train, y_train = Decoder.append_previous_n_samples(
            X_train,
            y_train,
            n=n
        )
        X_val, y_val = Decoder.append_previous_n_samples(
            X_val,
            y_val,
            n=n
        )
        return X_train, y_train, X_val, y_val

    def fit_model(self, model, X_train, y_train):

        if self.TRAIN_VAL_SPLIT is True:
            X_train, X_val, y_train, y_val = \
                model_selection.train_test_split(
                    X_train, y_train, train_size=0.7, shuffle=False)

            if y_train.sum() == 0 or y_val.sum(0) == 0:
                raise Decoder.ClassMissingException

            if type(model) is xgboost.sklearn.XGBClassifier:
                classes_weights = class_weight.compute_sample_weight(
                    class_weight='balanced',
                    y=y_train
                )

                model.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)],
                    early_stopping_rounds=7, sample_weight=classes_weights,
                    verbose=self.VERBOSE, eval_metric="logloss")
            else:
                # might be necessary to adapt for other classifiers
                model.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)])
        else:

            # check for LDA; and apply rebalancing
            if type(model) is discriminant_analysis.LinearDiscriminantAnalysis:
                X_train, y_train = self.ros.fit_resample(X_train, y_train)

            if type(model) is xgboost.sklearn.XGBClassifier:
                model.fit(X_train, y_train, eval_metric="logloss")  # to avoid warning
            else:
                model.fit(X_train, y_train)

        return model

    def eval_model(
        self,
        model_train,
        X_train,
        X_test,
        y_train,
        y_test,
        cv_res : Type[CV_res]
    ) -> Type[CV_res]:

        if self.save_coef:
            cv_res.coef.append(model_train.coef_)

        y_test_pr = model_train.predict(X_test)
        y_train_pr = model_train.predict(X_train)

        sc_te = self.eval_method(y_test, y_test_pr)
        sc_tr = self.eval_method(y_train, y_train_pr)

        if self.threshold_score is True:
            if sc_tr < 0:
                sc_tr = 0
            if sc_te < 0:
                sc_te = 0

        if self.get_movement_detection_rate is True:
            self._set_movement_detection_rates(
                y_test,
                y_test_pr,
                y_train,
                y_train_pr,
                cv_res
            )

        cv_res.score_train.append(sc_tr)
        cv_res.score_test.append(sc_te)
        cv_res.X_train.append(X_train)
        cv_res.X_test.append(X_test)
        cv_res.y_train.append(y_train)
        cv_res.y_test.append(y_test)
        cv_res.y_train_pr.append(y_train_pr)
        cv_res.y_test_pr.append(y_test_pr)
        return cv_res

    def _set_movement_detection_rates(
        self,
        y_test: np.ndarray,
        y_test_pr: np.ndarray,
        y_train: np.ndarray,
        y_train_pr: np.ndarray,
        cv_res : Type[CV_res],
    ) -> Type[CV_res]:

        mov_detection_rate, fpr, tpr = self.calc_movement_detection_rate(
            y_test,
            y_test_pr,
            self.mov_detection_threshold,
            self.min_consequent_count
        )

        cv_res.mov_detection_rates_test.append(mov_detection_rate)
        cv_res.tprate_test.append(tpr)
        cv_res.fprate_test.append(fpr)

        mov_detection_rate, fpr, tpr = self.calc_movement_detection_rate(
            y_train,
            y_train_pr,
            self.mov_detection_threshold,
            self.min_consequent_count
        )

        cv_res.mov_detection_rates_train.append(mov_detection_rate)
        cv_res.tprate_train.append(tpr)
        cv_res.fprate_train.append(fpr)

        return cv_res
    
    def wrapper_model_train(
        self,
        X_train,
        y_train,
        X_test = None,
        y_test = None,
        cv_res: Type[CV_res] = CV_res(),
        return_fitted_model_only : bool = False
    ):

        model_train = clone(self.model)
        if self.STACK_FEATURES_N_SAMPLES is True:
            if X_train is None:
                X_train, y_train = Decoder.append_previous_n_samples(
                    X_train, y_train, self.time_stack_n_samples_
                )
            else:
                X_train, y_train, X_test, y_test = Decoder.append_samples_val(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    n=self.time_stack_n_samples
                )
            

        if y_train.sum() == 0 or y_test.sum() == 0:  # only one class present
            raise Decoder.ClassMissingException

        if self.RUN_BAY_OPT is True:
            model_train = self.bay_opt_wrapper(model_train, X_train, y_train)

        # fit model
        model_train = self.fit_model(model_train, X_train, y_train)

        if return_fitted_model_only is True:
            return model_train

        cv_res = self.eval_model(
            model_train,
            X_train,
            X_test,
            y_train,
            y_test,
            cv_res
        )

        return cv_res

    def run_CV(self, data, label):
        """Evaluate model performance on the specified cross validation.
        If no data and label is specified, use whole feature class attributes.

        Parameters
        ----------
        data (np.ndarray):
            data to train and test with shape samples, features
        label (np.ndarray):
            label to train and test with shape samples, features
        """

        def split_data(data):
            if self.cv_method == "NonShuffledTrainTestSplit":
                # unfortunately lot of overhead since a non shuffle index base split 
                # is not supported in sklearn
                cv_single_tr_te_split =  model_selection.check_cv(
                    cv=[
                        model_selection.train_test_split(
                            np.arange(data.shape[0]),
                            test_size=0.3,
                            shuffle=False
                        )
                    ]
                )
                for train_index, test_index in cv_single_tr_te_split.split():
                    yield train_index, test_index
            else:
                for train_index, test_index in self.cv_method.split(data):
                    yield train_index, test_index

        cv_res = self.init_cv_res()

        if self.use_nested_cv is True:
            cv_res_inner = self.init_cv_res()

        for train_index, test_index in split_data(data):
            X_train, y_train = data[train_index, :], label[train_index]
            X_test, y_test = data[test_index], label[test_index]
            try:
                cv_res = self.wrapper_model_train(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    cv_res
                )
            except Decoder.ClassMissingException:
                continue

            if self.use_nested_cv is True:
                data_inner = data[train_index]
                label_inner = label[train_index]
                for train_index_inner, test_index_inner in split_data(data_inner):
                    X_train_inner = data_inner[train_index_inner, :]
                    y_train_inner = label_inner[train_index_inner]
                    X_test_inner = data_inner[test_index_inner]
                    y_test_inner = label_inner[test_index_inner]
                    try:
                        cv_res_inner = self.wrapper_model_train(
                            X_train_inner,
                            y_train_inner,
                            X_test_inner,
                            y_test_inner,
                            cv_res_inner
                        )
                    except Decoder.ClassMissingException:
                        continue

        self.cv_res = cv_res
        if self.use_nested_cv is True:
            self.cv_res_inner = cv_res_inner

    def bay_opt_wrapper(self, model_train, X_train, y_train):
        """Run bayesian optimization and test best params to model_train
        Save best params into self.best_bay_opt_params
        """

        X_train_bo, X_test_bo, y_train_bo, y_test_bo = \
            model_selection.train_test_split(
                X_train, y_train, train_size=0.7, shuffle=False)

        if y_train_bo.sum() == 0 or y_test_bo.sum() == 0:
            print("could not start Bay. Opt. with no labels > 0")
            raise Decoder.ClassMissingException

        params_bo = self.run_Bay_Opt(
            X_train_bo,
            y_train_bo,
            X_test_bo,
            y_test_bo,
            rounds=10
        )

        # set bay. opt. obtained best params to model
        params_bo_dict = {}
        for i in range(len(params_bo)):
            setattr(
                model_train,
                self.bay_opt_param_space[i].name,
                params_bo[i]
            )
            params_bo_dict[self.bay_opt_param_space[i].name] = params_bo[i]

        self.best_bay_opt_params.append(params_bo_dict)

        return model_train

    def run_Bay_Opt(self,
        X_train,
        y_train,
        X_test,
        y_test,
        rounds=30,
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="sampling",
        initial_point_generator="lhs"
    ):
        """Run skopt bayesian optimization
        skopt.Optimizer:
        https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer

        example:
        https://scikit-optimize.github.io/stable/auto_examples/ask-and-tell.html#sphx-glr-auto-examples-ask-and-tell-py

        Special attention needs to be made with the run_CV output,
        some metrics are minimized (MAE), some are maximized (r^2)

        Parameters
        ----------
        X_train: np.ndarray
        y_train: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        rounds : int, optional
            optimizing rounds, by default 10
        base_estimator : str, optional
            surrogate model, used as optimization function instead of cross validation, by default "GP"
        acq_func : str, optional
            function to minimize over the posterior distribution, by default "EI"
        acq_optimizer : str, optional
            method to minimize the acquisition function, by default "sampling"
        initial_point_generator : str, optional
            sets a initial point generator, by default "lhs"

        Returns
        -------
        skopt result parameters
        """

        def get_f_val(model_bo):

            try:
                model_bo = self.fit_model(model_bo, X_train, y_train)
            except Decoder.ClassMissingException:
                pass

            return self.eval_method(y_test, model_bo.predict(X_test))

        opt = Optimizer(
            self.bay_opt_param_space,
            base_estimator=base_estimator,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            initial_point_generator=initial_point_generator
        )

        for _ in range(rounds):
            next_x = opt.ask()
            # set model values
            model_bo = clone(self.model)
            for i in range(len(next_x)):
                setattr(model_bo, self.bay_opt_param_space[i].name, next_x[i])
            f_val = get_f_val(model_bo)
            res = opt.tell(next_x, f_val)
            if self.VERBOSE:
                print(f_val)

        # res is here automatically appended by skopt
        return res.x

    def save(self, feature_path: str, feature_file: str, str_save_add=None) -> None:
        """Save decoder object to pickle
        """

        # why is the decoder not saved to a .json?

        if str_save_add is None:
            PATH_OUT = os.path.join(feature_path, feature_file, feature_file + "_ML_RES.p")
        else:
            PATH_OUT = os.path.join(feature_path, feature_file, feature_file +
                                    "_" + str_save_add + "_ML_RES.p")

        print("model being saved to: " + str(PATH_OUT))
        with open(PATH_OUT, 'wb') as output:  # Overwrites any existing file.
            cPickle.dump(self, output)
