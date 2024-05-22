import copy
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from xgboost.callback import (
    CallbackContainer,
    EarlyStopping,
    EvaluationMonitor,
    TrainingCallback,
)
from xgboost.core import (
    Booster,
    DMatrix,
    Metric,
    Objective,
    _LIB,
    _check_call,
    ctypes,
)

from xgboost.training import (
    _configure_custom_metric,   
)


def scale_update(
    self, dtrain: DMatrix, iteration: int, fobj: Optional[Objective] = None, \
        scores: Optional[np.ndarray] = None, scale: float = 0.0) -> None:
    """Update for one iteration, with objective function calculated
    internally.  This function should not be called directly by users.

    Parameters
    ----------
    dtrain :
        Training data.
    iteration :
        Current iteration number.
    fobj :
        Customized objective function.

    """
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f"invalid training matrix: {type(dtrain).__name__}")
    self._assign_dmatrix_features(dtrain)

    if fobj is None:
        _check_call(
            _LIB.XGBoosterUpdateOneIter(
                self.handle, ctypes.c_int(iteration), dtrain.handle
            )
        )
    else:
        # pred = self.predict(dtrain, output_margin=True, training=True)
        if scores is not None:
            pred = self.predict(dtrain, output_margin=True, training=True) + scores*scale
        else:
            pred = self.predict(dtrain, output_margin=True, training=True)
        # scores = scores*scale + 0.5
        # if iteration==0:
        #     pred=scores
        # else:
        #     pred = scores + self.predict(dtrain, output_margin=True, training=True, iteration_range=(1, iteration))
        grad, hess = fobj(pred, dtrain, scores, iteration)
        self.boost(dtrain, iteration=iteration, grad=grad, hess=hess)
        

def scale_train(
    params: Dict[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    evals: Optional[Sequence[Tuple[DMatrix, str]]] = None,
    obj: Optional[Objective] = None,
    feval: Optional[Metric] = None,
    maximize: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
    evals_result: Optional[TrainingCallback.EvalsLog] = None,
    verbose_eval: Optional[Union[bool, int]] = True,
    xgb_model: Optional[Union[str, os.PathLike, Booster, bytearray]] = None,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    custom_metric: Optional[Metric] = None,
    scores: Optional[np.ndarray] = None,
    scale: float = 0.0,
) -> Booster:
    """Train a booster with given parameters.

    Parameters
    ----------
    params :
        Booster params.
    dtrain :
        Data to be trained.
    num_boost_round :
        Number of boosting iterations.
    evals :
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    obj
        Custom objective function.  See :doc:`Custom Objective
        </tutorials/custom_metric_obj>` for details.
    feval :
        .. deprecated:: 1.6.0
            Use `custom_metric` instead.
    maximize :
        Whether to maximize feval.
    early_stopping_rounds :
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        Requires at least one item in **evals**.
        The method returns the model from the last iteration (not the best one).  Use
        custom callback or model slicing if the best model is desired.
        If there's more than one item in **evals**, the last entry will be used for early
        stopping.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
        If early stopping occurs, the model will have two additional fields:
        ``bst.best_score``, ``bst.best_iteration``.
    evals_result :
        This dictionary stores the evaluation results of all the items in watchlist.

        Example: with a watchlist containing
        ``[(dtest,'eval'), (dtrain,'train')]`` and
        a parameter containing ``('eval_metric': 'logloss')``,
        the **evals_result** returns

        .. code-block:: python

            {'train': {'logloss': ['0.48253', '0.35953']},
             'eval': {'logloss': ['0.480385', '0.357756']}}

    verbose_eval :
        Requires at least one item in **evals**.
        If **verbose_eval** is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If **verbose_eval** is an integer then the evaluation metric on the validation set
        is printed at every given **verbose_eval** boosting stage. The last boosting stage
        / the boosting stage found by using **early_stopping_rounds** is also printed.
        Example: with ``verbose_eval=4`` and at least one item in **evals**, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    xgb_model :
        Xgb model to be loaded before training (allows training continuation).
    callbacks :
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.

        .. note::

           States in callback are not preserved during training, which means callback
           objects can not be reused for multiple training sessions without
           reinitialization or deepcopy.

        .. code-block:: python

            for params in parameters_grid:
                # be sure to (re)initialize the callbacks before each run
                callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
                xgboost.train(params, Xy, callbacks=callbacks)

    custom_metric:

        .. versionadded 1.6.0

        Custom metric function.  See :doc:`Custom Metric </tutorials/custom_metric_obj>`
        for details.

    Returns
    -------
    Booster : a trained booster model
    """

    callbacks = [] if callbacks is None else copy.copy(list(callbacks))
    metric_fn = _configure_custom_metric(feval, custom_metric)
    evals = list(evals) if evals else []

    bst = Booster(params, [dtrain] + [d[0] for d in evals], model_file=xgb_model)
    start_iteration = 0

    if verbose_eval:
        verbose_eval = 1 if verbose_eval is True else verbose_eval
        callbacks.append(EvaluationMonitor(period=verbose_eval))
    if early_stopping_rounds:
        callbacks.append(EarlyStopping(rounds=early_stopping_rounds, maximize=maximize))
    cb_container = CallbackContainer(
        callbacks,
        metric=metric_fn,
        # For old `feval` parameter, the behavior is unchanged.  For the new
        # `custom_metric`, it will receive proper prediction result when custom objective
        # is not used.
        output_margin=callable(obj) or metric_fn is feval,
    )

    bst = cb_container.before_training(bst)

    for i in range(start_iteration, num_boost_round):
        if cb_container.before_iteration(bst, i, dtrain, evals):
            break
        bst.update(dtrain, iteration=i, fobj=obj, scores=scores, scale=scale)
        if cb_container.after_iteration(bst, i, dtrain, evals):
            break

    bst = cb_container.after_training(bst)

    if evals_result is not None:
        evals_result.update(cb_container.history)

    # Copy to serialise and unserialise booster to reset state and free
    # training memory
    return bst.copy()