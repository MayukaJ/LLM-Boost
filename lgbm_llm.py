import os
import numpy as np
import pandas as pd
# import zero
from sklearn.metrics import f1_score, accuracy_score
import random
from llm_boost_utils import load_tabular_data, predict_lgbm, \
    softprob_obj_lgbm, detect_and_encode_categorical
import optuna
import argparse
import lightgbm as lgb
from LGBMscale.utils import scale_update, scale_train
lgb.basic.Booster.update = scale_update
lgb.train = scale_train

import logging

seed = 0
random.seed(seed)
np.random.seed(seed)

class CustomLogger:
    def init(self):
        self.logger = logging.getLogger('lightgbm_custom')
        self.logger.setLevel(logging.ERROR)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        # Suppress warnings by not doing anything
        pass

    def error(self, message):
        self.logger.error(message)
l = CustomLogger()
l.init()
lgb.register_logger(l)

parser = argparse.ArgumentParser(description="Run Optuna")
parser.add_argument("--data_path", default='./data/Adult_flan/', type=str, help="Path to dataset")
parser.add_argument("--train_size", default="-1", type=int, help="train_size")
parser.add_argument("--test_size", default="-1", type=int, help="test_size")
parser.add_argument("--val_size", default="0.5", type=float, help="val_size")
parser.add_argument("--cv_folds", default="1", type=int, help="number of cv_folds")
parser.add_argument("--stratified", default="0", type=bool, help="stratify dataset")
parser.add_argument('--stack', action='store_true')
args = parser.parse_args()

data = load_tabular_data(data_paths=[args.data_path], train_size=args.train_size,
                         test_size=args.test_size, num_exp=1, cv_folds=args.cv_folds,
                         stratified=False, val_size=args.val_size, stack=args.stack)
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

        df_list, _ = detect_and_encode_categorical([train_x, test_x, val_x], encoding_type="category")
        train_x, test_x, val_x = df_list[0], df_list[1], df_list[2]
        
        try:
            train_y = np.argmax(train_y, axis=1)
            test_y = np.argmax(test_y, axis=1)
            val_y = np.argmax(val_y, axis=1)
            train_m = lgb.Dataset(train_x, train_y, params={'verbosity': -1})
            # test_m = lgb.Dataset(test_x, test_y, params={'verbosity': -1})
            # val_m = lgb.Dataset(val_x, val_y, params={'verbosity': -1})
            
            model = lgb.train(params,
                            train_m,
                            num_boost_round=num_boost_round,
                            scores=scores_train,
                            scale=scale,
                            )
            y_pred = predict_lgbm(model, 
                            test_x,
                            scores=scores_test,
                            scale=scale,
                            )
            acc.append(accuracy_score(test_y, y_pred))
            y_pred_val = predict_lgbm(model, 
                            val_x,
                            scores=scores_val,
                            scale=scale,
                            )
            val_acc.append(accuracy_score(val_y, y_pred_val))
            
        except:
            pass
    
    return np.mean(acc), np.mean(val_acc)


def objective(trial, test_scores):
    params = {
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    params['objective'] = softprob_obj_lgbm
    params['num_class'] = N_CLASSES
    params['verbosity'] = -1
    params["scale"] = 0.0
    params["num_boost_round"] = 100
    
    test_acc, val_acc = train(data, params)
    print("val score: %f\t test score:%f"%(val_acc, test_acc))
    test_scores.append(test_acc)
    return val_acc

def objective2(trial, test_scores, params):
    scale = trial.suggest_float("scale", 1e-4, 1e4, log=True)
    params['objective'] = softprob_obj_lgbm
    params['num_class'] = N_CLASSES
    params['verbosity'] = -1
    params["scale"] = scale
    params["num_boost_round"] = 100
    
    test_acc, val_acc = train(data, params)
    print("val score: %f\t test score:%f"%(val_acc, test_acc))
    test_scores.append(test_acc)
    return val_acc

if __name__ == "__main__":
    test_scores = []
    func = lambda trial: objective(trial, test_scores)
    study = optuna.create_study(direction='maximize')
    study.optimize(func, n_trials=100)
    best_trial = study.best_trial
    best_test = test_scores[best_trial.number]
    
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
    
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print('Best hyperparameters:', study.best_params)
    print('Best Standard Val Acc:', study.best_value)
    print('Best Standard Test Acc:', best_test)
    
    print('Best Scaling Value:', study2.best_params)
    print('Best Fusion Val Acc:', study2.best_value)
    print('Best Fusion Test Acc:', best_test2)
    
    llm_acc = []
    for i in range(len(data[0])):
        train_x, test_x, val_x, train_y, test_y, val_y, scores_train, scores_test, scores_val = data[0][i]
        llm_acc.append(accuracy_score(np.argmax(test_y, axis=1), np.argmax(scores_test, axis=1)))
    
    print('LLM Acc:', np.mean(llm_acc))
    