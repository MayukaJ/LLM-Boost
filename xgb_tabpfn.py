import os
import numpy as np
import pandas as pd
# import zero
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import scipy.special
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
# from xgboost import XGBRegressor, XGBClassifier
from XGBscale.utils import scale_update, scale_train
xgb.train = scale_train
xgb.core.Booster.update = scale_update
import random
from typing import Tuple, Optional
from llm_boost_utils import load_tabpfn_data, softprob_obj, predict, merror, append_line_to_csv
import optuna
import argparse
import scipy.special
import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)

seed = 0
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description="Run Optuna")
parser.add_argument("--data_path", default='./data/Adult/prototypes-synthetic-performance-0/', type=str, help="Path to dataset")
parser.add_argument("--train_size", default="-1", type=int, help="train_size")
parser.add_argument("--test_size", default="-1", type=int, help="test_size")
# parser.add_argument("--pfn_size", default="10", type=int, help="pfn_size")
parser.add_argument("--val_size", default="0.5", type=float, help="val_size")
# parser.add_argument("--num_exp", default="1", type=int, help="num_exp")
parser.add_argument("--cv_folds", default="5", type=int, help="number of cv_folds")
parser.add_argument("--stratified", action='store_true')
parser.add_argument("--use_standard", action="store_true", help="Use standard model")
parser.add_argument("--use_caafe", action="store_true", help="Use CAAFE data")
parser.add_argument("--data_id", default="31", type=int, help="data_id")
parser.add_argument('--stack', action='store_true')
parser.add_argument("--seed", default="0", type=int, help="seed")
args = parser.parse_args()


data = load_tabpfn_data(data_paths=[args.data_path], train_size=args.train_size,
                         test_size=args.test_size, num_exp=1, 
                         cv_folds=args.cv_folds, stratified=args.stratified, val_size=args.val_size, use_oml=False, data_id=args.data_id,
                         stack=args.stack, use_caafe=args.use_caafe, seed=args.seed)

data2 = load_tabpfn_data(data_paths=[args.data_path], train_size=args.train_size,
                         test_size=args.test_size, num_exp=1, 
                         cv_folds=args.cv_folds, stratified=args.stratified, val_size=args.val_size, use_oml=False, data_id=args.data_id,
                         stack=True, use_caafe=args.use_caafe, seed=args.seed)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
from llm_boost_utils import N_CLASSES

def train(data, params):
    acc = []
    val_acc = []
    num_boost_round = params.pop('num_boost_round')
    scale = params.pop('scale')
    if scale < 1e-5:
        scale = 0.0

    for i in range(len(data[0])):
        train_x, test_x, val_x, train_y, test_y, val_y, scores_train, scores_test, scores_val = data[0][i]
        
        try:
            train_m = xgb.DMatrix(train_x, train_y, enable_categorical=True)
            test_m = xgb.DMatrix(test_x, test_y, enable_categorical=True)
            val_m = xgb.DMatrix(val_x, val_y, enable_categorical=True)
            
            model = xgb.train(params,
                            train_m,
                            num_boost_round=num_boost_round,
                            obj=softprob_obj,
                            custom_metric=merror,
                            scores=scores_train,
                            scale=scale,
                            )
            y_pred = predict(model, 
                            test_m,
                            num_boost_round, 
                            scores_test,
                            scale=scale,
                            )
            acc.append(roc_auc_score(test_y, y_pred, multi_class='ovr'))
            y_pred_val = predict(model, 
                            val_m,
                            num_boost_round,
                            scores_val,
                            scale=scale,
                            )
            val_acc.append(roc_auc_score(val_y, y_pred_val, multi_class='ovr'))
            
        except:
            pass
    
    return np.mean(acc), np.mean(val_acc)


def objective(trial, test_scores, data):
    params = {
        # "num_boost_round": trial.suggest_int("num_boost_round", 1, 20), #2000
        "eta": trial.suggest_float("eta", 1e-5, 1, log=True), #learning rate
        "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True), #L2 penalty
        "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True), #L1 penalty
        "gamma": trial.suggest_float("gamma", 1e-8, 1e2, log=True), #min split loss
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
    }
    
    params["num_class"] = N_CLASSES
    params["disable_default_eval_metric"] = True
    params["scale"] = 0.0
    params["num_boost_round"] = 20
    
    test_acc, val_acc = train(data, params)
    print("val score: %f\t test score:%f"%(val_acc, test_acc))
    test_scores.append(test_acc)
    return val_acc


def objective2(trial, test_scores, params):
    scale = trial.suggest_float("scale", 1e-4, 1e4, log=True)
    params["num_class"] = N_CLASSES
    params["disable_default_eval_metric"] = True
    params["scale"] = scale
    params["num_boost_round"] = 20
    
    test_acc, val_acc = train(data, params)
    print("val score: %f\t test score:%f"%(val_acc, test_acc))
    test_scores.append(test_acc)
    return val_acc


if __name__ == "__main__":
    test_scores = []
    func = lambda trial: objective(trial, test_scores, data)
    study = optuna.create_study(direction='maximize')
    study.optimize(func, n_trials=100)
    # best_trial = study.best_trial
    # best_test = test_scores[best_trial.number]
    
    if not args.stack:
        test_scores2 = []
        func2 = lambda trial: objective2(trial, test_scores2, study.best_params)
        study2 = optuna.create_study(direction='maximize')
        study2.enqueue_trial(
            {
                "scale": float(1e-6),
            }
        )
        study2.enqueue_trial(
            {
                "scale": float(1e6),
            }
        )
        study2.optimize(func2, n_trials=30)
        best_trial2 = study2.best_trial
        best_test2 = test_scores2[best_trial2.number]
    
    test_scores3 = []
    func3 = lambda trial: objective(trial, test_scores3, data2)
    study3 = optuna.create_study(direction='maximize')
    study3.optimize(func3, n_trials=130)
    best_trial3 = study3.best_trial
    best_test3 = test_scores3[best_trial3.number]
    
    test_scores4 = []
    func4 = lambda trial: objective(trial, test_scores4, data)
    study4 = optuna.create_study(direction='maximize')
    study4.enqueue_trial(study.best_params)
    study4.optimize(func4, n_trials=30)
    best_trial4 = study4.best_trial
    best_test4 = test_scores4[best_trial4.number]
    
    test_scores5 = []
    func5 = lambda trial: objective2(trial, test_scores5, study4.best_params)
    study5 = optuna.create_study(direction='maximize')
    study5.enqueue_trial(
        {
            "scale": float(1e-6),
        }
    )
    study5.enqueue_trial(
        {
            "scale": float(1e6),
        }
    )
    study5.optimize(func5, n_trials=2)
    best_trial5 = study5.best_trial
    best_test5 = test_scores5[best_trial5.number]
    
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print('Best hyperparameters:', study.best_params)
    print('Best Standard Val Acc:', study.best_value)
    print('Best Standard Test Acc:', best_test4)
    
    if not args.stack:
        print('Best Scaling Value:', study2.best_params)
        print('Best Fusion Val Acc:', study2.best_value)
        print('Best Fusion Test Acc:', best_test2)
    
    print('Best Scaling Value:', study3.best_params)
    print('Best Fusion Val Acc:', study3.best_value)
    print('Best Fusion Test Acc:', best_test3)
    
    llm_acc = []
    for i in range(len(data[0])):
        train_x, test_x, val_x, train_y, test_y, val_y, scores_train, scores_test, scores_val = data[0][i]
        scores_test = scipy.special.softmax(scores_test, axis=1)
        if N_CLASSES <= 2:
            scores_test = scores_test[:,1]
        llm_acc.append(roc_auc_score(test_y, scores_test, multi_class='ovr'))
    
    print('LLM Acc:', np.mean(llm_acc))
    llm_acc = np.mean(llm_acc)
    
    print(study5.trials[0].value)
    print(study5.trials[1].value)
    if study5.trials[0].value >= study5.trials[1].value:
        best = best_test4
    else:
        best = llm_acc
    
    file_path = 'tabpfn_caafe.csv'
    
    # data_to_append = [args.data_path.split("/")[-2], args.train_size, args.seed, best_test4, llm_acc, best, best_test3, best_test2]
    data_to_append = [args.data_id, args.seed, best_test4, llm_acc, best, best_test3, best_test2]

    
    append_line_to_csv(file_path, data_to_append)

    