import copy
import json
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from lightgbm import callback
from lightgbm.basic import (
    Booster,
    Dataset,
    LightGBMError,
    _choose_param_value,
    _ConfigAliases,
    _InnerPredictor,
    _LGBM_BoosterEvalMethodResultType,
    _LGBM_BoosterEvalMethodResultWithStandardDeviationType,
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_CustomObjectiveFunction,
    _LGBM_EvalFunctionResultType,
    _LGBM_FeatureNameConfiguration,
    _log_warning,
    _LIB,
    _safe_call,
    ctypes,
)


_LGBM_CustomMetricFunction = Union[
    Callable[
        [np.ndarray, Dataset],
        _LGBM_EvalFunctionResultType,
    ],
    Callable[
        [np.ndarray, Dataset],
        List[_LGBM_EvalFunctionResultType],
    ],
]


def scale_update(
    self,
    train_set: Optional[Dataset] = None,
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None,
    scores: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> bool:
    """Update Booster for one iteration.

    Parameters
    ----------
    train_set : Dataset or None, optional (default=None)
        Training data.
        If None, last training data is used.
    fobj : callable or None, optional (default=None)
        Customized objective function.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            train_data : Dataset
                The training dataset.
            grad : numpy 1-D array or numpy 2-D array (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of preds for each sample point.
            hess : numpy 1-D array or numpy 2-D array (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of preds for each sample point.

        For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.

    Returns
    -------
    is_finished : bool
        Whether the update was successfully finished.
    """
    # need reset training data
    if train_set is None and self.train_set_version != self.train_set.version:
        train_set = self.train_set
        is_the_same_train_set = False
    else:
        is_the_same_train_set = train_set is self.train_set and self.train_set_version == train_set.version
    if train_set is not None and not is_the_same_train_set:
        if not isinstance(train_set, Dataset):
            raise TypeError(f"Training data should be Dataset instance, met {type(train_set).__name__}")
        if train_set._predictor is not self.__init_predictor:
            raise LightGBMError("Replace training data failed, " "you should use same predictor for these data")
        self.train_set = train_set
        _safe_call(
            _LIB.LGBM_BoosterResetTrainingData(
                self._handle,
                self.train_set.construct()._handle,
            )
        )
        self.__inner_predict_buffer[0] = None
        self.train_set_version = self.train_set.version
    is_finished = ctypes.c_int(0)
    if fobj is None:
        if self.__set_objective_to_none:
            raise LightGBMError("Cannot update due to null objective function.")
        _safe_call(
            _LIB.LGBM_BoosterUpdateOneIter(
                self._handle,
                ctypes.byref(is_finished),
            )
        )
        self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
        return is_finished.value == 1
    else:
        if not self.__set_objective_to_none:
            self.reset_parameter({"objective": "none"}).__set_objective_to_none = True
        # grad, hess = fobj(self.__inner_predict(0), self.train_set)
        if scores is not None:
            pred = self.__inner_predict(0) + scale*scores
        else:
            pred = self.__inner_predict(0)
        grad, hess = fobj(pred, self.train_set)
        return self.__boost(grad, hess)


def scale_train(
    params: Dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[_LGBM_CustomMetricFunction, List[_LGBM_CustomMetricFunction]]] = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    feature_name: _LGBM_FeatureNameConfiguration = "auto",
    categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    scores: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> Booster:
    """Perform the training with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    valid_sets : list of Dataset, or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of str, or None, optional (default=None)
        Names of ``valid_sets``.
    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, eval_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                The predicted values.
                For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                If custom objective function is used, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            eval_data : Dataset
                A ``Dataset`` to evaluate.
            eval_name : str
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        To ignore the default metric corresponding to the used objective,
        set the ``metric`` parameter to the string ``"None"`` in ``params``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        Floating point numbers in categorical features will be rounded towards 0.
    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.

    Note
    ----
    A custom objective function can be provided for the ``objective`` parameter.
    It should accept two parameters: preds, train_data and return (grad, hess).

        preds : numpy 1-D array or numpy 2-D array (for multi-class task)
            The predicted values.
            Predicted values are returned before any transformation,
            e.g. they are raw margin instead of probability of positive class for binary task.
        train_data : Dataset
            The training dataset.
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.

    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
    and grad and hess should be returned in the same format.

    Returns
    -------
    booster : Booster
        The trained Booster model.
    """
    if not isinstance(train_set, Dataset):
        raise TypeError(f"train() only accepts Dataset object, train_set has type '{type(train_set).__name__}'.")

    if num_boost_round <= 0:
        raise ValueError(f"num_boost_round must be greater than 0. Got {num_boost_round}.")

    if isinstance(valid_sets, list):
        for i, valid_item in enumerate(valid_sets):
            if not isinstance(valid_item, Dataset):
                raise TypeError(
                    "Every item in valid_sets must be a Dataset object. "
                    f"Item {i} has type '{type(valid_item).__name__}'."
                )

    # create predictor first
    params = copy.deepcopy(params)
    params = _choose_param_value(
        main_param_name="objective",
        params=params,
        default_value=None,
    )
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = "none"
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
    params["num_iterations"] = num_boost_round
    # setting early stopping via global params should be possible
    params = _choose_param_value(
        main_param_name="early_stopping_round",
        params=params,
        default_value=None,
    )
    if params["early_stopping_round"] is None:
        params.pop("early_stopping_round")
    first_metric_only = params.get("first_metric_only", False)

    predictor: Optional[_InnerPredictor] = None
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor.from_model_file(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = _InnerPredictor.from_booster(booster=init_model, pred_parameter=dict(init_model.params, **params))

    if predictor is not None:
        init_iteration = predictor.current_iteration()
    else:
        init_iteration = 0

    train_set._update_params(params)._set_predictor(predictor).set_feature_name(feature_name).set_categorical_feature(
        categorical_feature
    )

    is_valid_contain_train = False
    train_data_name = "training"
    reduced_valid_sets = []
    name_valid_sets = []
    if valid_sets is not None:
        if isinstance(valid_sets, Dataset):
            valid_sets = [valid_sets]
        if isinstance(valid_names, str):
            valid_names = [valid_names]
        for i, valid_data in enumerate(valid_sets):
            # reduce cost for prediction training data
            if valid_data is train_set:
                is_valid_contain_train = True
                if valid_names is not None:
                    train_data_name = valid_names[i]
                continue
            reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
            if valid_names is not None and len(valid_names) > i:
                name_valid_sets.append(valid_names[i])
            else:
                name_valid_sets.append(f"valid_{i}")
    # process callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault("order", i - len(callbacks))
        callbacks_set = set(callbacks)

    if "early_stopping_round" in params:
        callbacks_set.add(
            callback.early_stopping(
                stopping_rounds=params["early_stopping_round"],  # type: ignore[arg-type]
                first_metric_only=first_metric_only,
                verbose=_choose_param_value(
                    main_param_name="verbosity",
                    params=params,
                    default_value=1,
                ).pop("verbosity")
                > 0,
            )
        )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, "before_iteration", False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter("order"))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter("order"))

    # construct booster
    try:
        booster = Booster(params=params, train_set=train_set)
        if is_valid_contain_train:
            booster.set_train_data_name(train_data_name)
        for valid_set, name_valid_set in zip(reduced_valid_sets, name_valid_sets):
            booster.add_valid(valid_set, name_valid_set)
    finally:
        train_set._reverse_update_params()
        for valid_set in reduced_valid_sets:
            valid_set._reverse_update_params()
    booster.best_iteration = 0

    # start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        for cb in callbacks_before_iter:
            cb(
                callback.CallbackEnv(
                    model=booster,
                    params=params,
                    iteration=i,
                    begin_iteration=init_iteration,
                    end_iteration=init_iteration + num_boost_round,
                    evaluation_result_list=None,
                )
            )

        booster.update(fobj=fobj, scores=scores, scale=scale)

        evaluation_result_list: List[_LGBM_BoosterEvalMethodResultType] = []
        # check evaluation result.
        if valid_sets is not None:
            if is_valid_contain_train:
                evaluation_result_list.extend(booster.eval_train(feval))
            evaluation_result_list.extend(booster.eval_valid(feval))
        try:
            for cb in callbacks_after_iter:
                cb(
                    callback.CallbackEnv(
                        model=booster,
                        params=params,
                        iteration=i,
                        begin_iteration=init_iteration,
                        end_iteration=init_iteration + num_boost_round,
                        evaluation_result_list=evaluation_result_list,
                    )
                )
        except callback.EarlyStopException as earlyStopException:
            booster.best_iteration = earlyStopException.best_iteration + 1
            evaluation_result_list = earlyStopException.best_score
            break
    booster.best_score = defaultdict(OrderedDict)
    for dataset_name, eval_name, score, _ in evaluation_result_list:
        booster.best_score[dataset_name][eval_name] = score
    if not keep_training_booster:
        booster.model_from_string(booster.model_to_string()).free_dataset()
    return booster


