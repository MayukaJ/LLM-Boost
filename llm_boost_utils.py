import os
import numpy as np
import pandas as pd
import csv
import fcntl
# import zero
from sklearn.metrics import f1_score, accuracy_score
import scipy.special
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import random
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import lightgbm as lgb
import openml
import math
# from CAAFE.caafe import data
# from CAAFE.caafe.preprocessing import make_datasets_numeric

N_CLASSES = 2

def append_line_to_csv(file_path, data):
    """
    Appends a line to the specified CSV file with file locking.
    
    :param file_path: Path to the CSV file.
    :param data: List of values to append as a new row.
    """
    with open(file_path, 'a', newline='') as file:
        # Lock the file for writing
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            writer = csv.writer(file)
            writer.writerow(data)
        finally:
            # Ensure the file is unlocked
            fcntl.flock(file, fcntl.LOCK_UN)


def softmax(x):
    '''Softmax function with x as input vector.'''
    # lim = 1e14
    # e = np.min(np.exp(x), lim)
    x = np.clip(x, -50, 50)
    e = np.exp(x)
    return e / np.sum(e)


def softprob_obj(predt: np.ndarray, data: xgb.DMatrix, scores: Optional[np.ndarray] = None, iteration=0):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    '''
    kRows = data.num_row()
    labels = data.get_label()
    # if iteration==0:
    #     if scores is not None:
    #         assert scores.shape == (kRows, N_CLASSES)
    #         predt = scores
    
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    assert predt.shape == (kRows, N_CLASSES)

    grad = np.zeros((kRows, N_CLASSES), dtype=float)
    hess = np.zeros((kRows, N_CLASSES), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= N_CLASSES
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = np.squeeze(g)
            hess[r, c] = h

    # Right now (XGBoost 1.0.0), reshaping is necessary
    # grad = grad.reshape((kRows * N_CLASSES, 1))
    # hess = hess.reshape((kRows * N_CLASSES, 1))
    return grad, hess


def softprob_obj_lgbm(predt: np.ndarray, data: lgb.Dataset):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    '''
    kRows = data.num_data()
    labels = data.get_label()
    
    if data.get_weight() == None:
        weights = np.ones((kRows, 1), dtype=float)
    elif data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    assert predt.shape == (kRows, N_CLASSES)

    grad = np.zeros((kRows, N_CLASSES), dtype=float)
    hess = np.zeros((kRows, N_CLASSES), dtype=float)

    eps = 1e-6

    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= N_CLASSES
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = np.squeeze(g)
            hess[r, c] = h
    return grad, hess


def predict_lgbm(booster: lgb.Booster, X: pd.DataFrame, scores: Optional[np.ndarray] = None,
                 scale: float = 0.0):
    '''A customized prediction function that converts raw prediction to
    target class.

    '''
    # Output margin means we want to obtain the raw prediction obtained from
    # tree leaf weight.
    kRows = len(X)
    # predt = booster.predict(X, output_margin=True)
    predt = booster.predict(X, raw_score=True)
    # predt = predt + scores
    if scores is not None:
        predt = predt + scores*scale
    # out = np.zeros(kRows)
    # for r in range(predt.shape[0]):
    #     # the class with maximum prob (not strictly prob as it haven't gone
    #     # through softmax yet so it doesn't sum to 1, but result is the same
    #     # for argmax).
    #     i = np.argmax(predt[r])
    #     out[r] = i
    # return out
    predt = scipy.special.softmax(predt, axis=1)
    if N_CLASSES <= 2:
        predt = predt[:,1]
    return predt


def predict(booster: xgb.Booster, X, num_boost_round,
            scores: Optional[np.ndarray] = None, scale: float = 0.0):
    '''A customized prediction function that converts raw prediction to
    target class.

    '''
    # Output margin means we want to obtain the raw prediction obtained from
    # tree leaf weight.
    kRows = X.num_row()
    predt = booster.predict(X, output_margin=True)
    # predt = booster.predict(X, output_margin=True, iteration_range=(1, num_boost_round))
    # predt = predt + scores
    if scores is not None:
        predt = predt + scores*scale# + 0.5
    # out = np.zeros(kRows)
    # for r in range(predt.shape[0]):
    #     # the class with maximum prob (not strictly prob as it haven't gone
    #     # through softmax yet so it doesn't sum to 1, but result is the same
    #     # for argmax).
    #     i = np.argmax(predt[r])
    #     out[r] = i
    # return out
    predt = scipy.special.softmax(predt, axis=1)
    if N_CLASSES <= 2:
        predt = predt[:,1]
    return predt


def merror(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    # Like custom objective, the predt is untransformed leaf weight when custom objective
    # is provided.

    # With the use of `custom_metric` parameter in train function, custom metric receives
    # raw input only when custom objective is also being used.  Otherwise custom metric
    # will receive transformed prediction.
    kRows = dtrain.num_row()
    assert predt.shape == (kRows, N_CLASSES)
    out = np.zeros(kRows)
    for r in range(predt.shape[0]):
        i = np.argmax(predt[r])
        out[r] = i

    assert y.shape == out.shape

    errors = np.zeros(kRows)
    errors[y != out] = 1.0
    return 'PyMError', np.sum(errors) / kRows


    
def detect_and_encode_categorical(df_list, encoding_type="label"):
    """
    Detects categorical columns in a DataFrame and encodes them numerically.

    Args:
        df (pandas.DataFrame): The DataFrame to encode.
        encoding_type (str, optional): The type of encoding to use. 
            Can be "label" for label encoding or "one-hot" for one-hot encoding.
            Defaults to "label".

    Returns:
        pandas.DataFrame: The DataFrame with encoded categorical columns.
    """

    # Define a function to check if a column is categorical
    
    def is_categorical(col):
        return col.dtype == 'object' or col.dtype == 'category' #col.nunique() < df.shape[0]

    df = pd.concat(df_list, axis=0)

    # Get categorical columns
    categorical_cols = [col for col in df.columns if is_categorical(df[col])]
    # return_cols = [df.columns.get_loc(col) for col in categorical_cols]

    # Perform encoding based on chosen type
    if encoding_type == "label":
    # Encode using LabelEncoder
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            for dff in df_list:
                dff[col] = encoder.transform(dff[col].astype(str))
    elif encoding_type == "ordinal":
        encoder = OrdinalEncoder()
        encoder.fit(df[categorical_cols].astype(str))
        for dff in df_list:
            dff[categorical_cols] = encoder.transform(dff[categorical_cols].astype(str))
    elif encoding_type == "category":
    # Encode using LabelEncoder
        for col in categorical_cols:
            for dff in df_list:
                dff[col] = dff[col].astype("category")
    elif encoding_type == "one-hot":
        # Encode using OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(df[categorical_cols])
        for dff in df_list:
            encoded_df = pd.DataFrame(encoder.transform(dff[categorical_cols]))
            encoded_df.columns = categorical_cols
            dff = pd.concat([dff.drop(categorical_cols, axis=1), encoded_df], axis=1)
    else:
        print(f"Invalid encoding type: {encoding_type}. Choose 'label' or 'one-hot'")
        return df_list, categorical_cols

    return df_list, categorical_cols


def load_tabular_data(data_paths, train_size=-1, test_size=-1, num_exp=1, cv_folds=5,
                      stratified=False, val_size=0.5, stack=False, seed=0):
    """Load tabular data."""
    
    train_path = os.path.join(data_paths[0], "train.csv")
    test_path = os.path.join(data_paths[0], "test.csv")
    train_x = pd.read_csv(train_path, index_col=0)
    test_x = pd.read_csv(test_path, index_col=0)
    # df = pd.read_csv(train_path, index_col=0)
    train_size = min(train_size, train_x.shape[0])
    label_cols = [col for col in train_x.columns if 'labels' in col]
    global N_CLASSES
    N_CLASSES = len(label_cols)
    output = []
    if len(data_paths) > 1:
        tmp_scores_train = pd.DataFrame()
        tmp_scores_test = pd.DataFrame()
        for i in range(N_CLASSES):
            tmp_scores_train['scores_%d'%i] = train_x['scores_%d'%i]
            tmp_scores_test['scores_%d'%i] = test_x['scores_%d'%i]
        
        for data_path in data_paths[1:]:
            train_path = os.path.join(data_path, "train.csv")
            test_path = os.path.join(data_path, "test.csv")  
            tmp_train_x = pd.read_csv(train_path, index_col=0)
            tmp_test_x = pd.read_csv(test_path, index_col=0)
            for i in range(N_CLASSES):
                tmp_scores_train['scores_%d'%i] = tmp_scores_train['scores_%d'%i]+tmp_train_x['scores_%d'%i]
                tmp_scores_test['scores_%d'%i] = tmp_scores_test['scores_%d'%i]+tmp_test_x['scores_%d'%i]
                
        for i in range(N_CLASSES):
            tmp_scores_train['scores_%d'%i] = tmp_scores_train['scores_%d'%i]/len(data_paths)
            tmp_scores_test['scores_%d'%i] = tmp_scores_test['scores_%d'%i]/len(data_paths)
            train_x['scores_%d'%i] = tmp_scores_train['scores_%d'%i]
            test_x['scores_%d'%i] = tmp_scores_test['scores_%d'%i]
    
    for iter in range(num_exp):
        
        tmp_train_x = train_x.copy()
        tmp_test_x = test_x.copy()
        
        if train_size>0:
            if stratified:
                one_hot_columns = ['labels_%d'%i for i in range(N_CLASSES)]
                tmp_train_x['label'] = tmp_train_x[one_hot_columns].idxmax(axis=1)
                min_stratum_size = tmp_train_x['label'].value_counts().min()
                sample_size = math.floor(train_size/N_CLASSES)
                if sample_size > min_stratum_size:
                    raise ValueError(f"Sample size {sample_size} exceeds the smallest stratum size {min_stratum_size}.")
                tmp_train_x = tmp_train_x.groupby('label', group_keys=False).apply(lambda x: x.sample(sample_size))
                tmp_train_x = tmp_train_x.drop(columns=['label'])
            else:
                tmp_train_x = tmp_train_x.sample(n=train_size, random_state=iter+seed).reset_index(drop=True)
        
        train_y = pd.DataFrame()
        for i in range(N_CLASSES):
            train_y['labels_%d'%i] = tmp_train_x['labels_%d'%i]
            tmp_train_x = tmp_train_x.drop(['labels_%d'%i], axis=1)
        train_y = np.array(train_y).astype('float32')
        scores_train = pd.DataFrame()
        for i in range(N_CLASSES):
            scores_train['scores_%d'%i] = tmp_train_x['scores_%d'%i]
            if not stack:
                tmp_train_x = tmp_train_x.drop(['scores_%d'%i], axis=1)
        scores_train = np.array(scores_train).astype('float32')
        scores_train = scores_train - np.max(scores_train, axis=1, keepdims=True)/2
        # scores_train = scipy.special.softmax(scores_train, axis=1)
    
        if test_size>0:
            if stratified:
                one_hot_columns = ['labels_%d'%i for i in range(N_CLASSES)]
                tmp_test_x['label'] = tmp_test_x[one_hot_columns].idxmax(axis=1)
                min_stratum_size = tmp_test_x['label'].value_counts().min()
                sample_size = math.floor(test_size/N_CLASSES)
                if sample_size > min_stratum_size:
                    raise ValueError(f"Sample size {sample_size} exceeds the smallest stratum size {min_stratum_size}.")
                tmp_test_x = tmp_test_x.groupby('label', group_keys=False).apply(lambda x: x.sample(sample_size))
                tmp_test_x = tmp_test_x.drop(columns=['label'])
            else:
                tmp_test_x = tmp_test_x.sample(n=test_size, random_state=iter+seed).reset_index(drop=True)
        
        test_y = pd.DataFrame()
        for i in range(N_CLASSES):
            test_y['labels_%d'%i] = tmp_test_x['labels_%d'%i]
            tmp_test_x = tmp_test_x.drop(['labels_%d'%i], axis=1)
        test_y = np.array(test_y).astype('float32')
        scores_test = pd.DataFrame()
        for i in range(N_CLASSES):
            scores_test['scores_%d'%i] = tmp_test_x['scores_%d'%i]
            if not stack:
                tmp_test_x = tmp_test_x.drop(['scores_%d'%i], axis=1)
        scores_test = np.array(scores_test).astype('float32')
        scores_test = scores_test - np.max(scores_test, axis=1, keepdims=True)/2
        # scores_test = scipy.special.softmax(scores_test, axis=1)
        inner = []
        for val_iter in range(cv_folds):
            if stratified:
                tmp_tmp_train_x, val_x, tmp_train_y, val_y, tmp_scores_train, scores_val = train_test_split(tmp_train_x, train_y, scores_train,
                                                                                            test_size=val_size, random_state=val_iter+seed, stratify=train_y)
            else:    
                tmp_tmp_train_x, val_x, tmp_train_y, val_y, tmp_scores_train, scores_val = train_test_split(tmp_train_x, train_y, scores_train,
                                                                                            test_size=val_size, random_state=val_iter+seed)
        
            inner.append((tmp_tmp_train_x, tmp_test_x, val_x, tmp_train_y, test_y, val_y,
                        tmp_scores_train, scores_test, scores_val))
        
        output.append(inner)
    
    return output



def load_tabpfn_data(data_paths, train_size=-1, test_size=-1, num_exp=1, cv_folds=5,
                      stratified=False, val_size=0.2, stack=False, use_oml=False, use_caafe=False, data_id=None, seed=0):
    """Load tabular data."""
    if use_oml:
        dataset = openml.datasets.get_dataset(data_id)
        train_x, train_y, cat_ind, col_names = dataset.get_data(dataset_format="dataframe")
        train_x, test_x = train_test_split(train_x, test_size=0.2, random_state=0)
        target_name = col_names[-1]
        # d = {True: 'c', False: 'q'}
        # cat_ind = [d[i] for i in cat_ind[:-1]]
    elif use_caafe:
        # cc_test_datasets_multiclass = data.load_all_data()
        # ds = cc_test_datasets_multiclass[data_id]
        # ds, train_x, test_x, _, _ = data.get_data_split(ds, seed=seed)
        # target_name = ds[4][-1]
        assert False, "Not implemented yet."
        
    else:
        train_path = os.path.join(data_paths[0], "train.csv")
        test_path = os.path.join(data_paths[0], "test.csv")
        train_x = pd.read_csv(train_path, index_col=0)
        test_x = pd.read_csv(test_path, index_col=0)
        target_name = 'y_temp'
        
    train_size = min(train_size, train_x.shape[0])
    global N_CLASSES
    output = []
    
    for iter in range(num_exp):
        tmp_train_x = train_x.copy()
        tmp_test_x = test_x.copy()
        
        if train_size>0:
            tmp_train_x = tmp_train_x.sample(n=train_size, random_state=iter+seed).reset_index(drop=True)
        if test_size>0:
            tmp_test_x = tmp_test_x.sample(n=test_size, random_state=iter+seed).reset_index(drop=True)
        tmp_train_y = tmp_train_x[target_name]
        tmp_train_x = tmp_train_x.drop([target_name], axis=1)
        tmp_test_y = tmp_test_x[target_name]
        tmp_test_x = tmp_test_x.drop([target_name], axis=1)
        
        df_list, cat_list = detect_and_encode_categorical([tmp_train_x, tmp_test_x], encoding_type="ordinal")
        tmp_train_x, tmp_test_x = df_list[0], df_list[1]
        le = LabelEncoder()
        le.fit(tmp_train_y)
        N_CLASSES = len(le.classes_)
        tmp_train_y = le.transform(tmp_train_y)
        tmp_test_y = le.transform(tmp_test_y)
        
        inner = []
        for val_iter in range(cv_folds):
            tmp_tmp_test_x = tmp_test_x.copy()
            tmp_tmp_train_x, val_x, tmp_tmp_train_y, val_y = train_test_split(tmp_train_x, tmp_train_y, test_size=val_size, random_state=val_iter+seed)
            if tmp_tmp_train_x.shape[0] >1000:
                pfn_x, _, pfn_y, _ = train_test_split(tmp_tmp_train_x, tmp_tmp_train_y, train_size=1000, random_state=iter+seed)
            else:
                pfn_x = tmp_tmp_train_x
                pfn_y = tmp_tmp_train_y
            tabfpn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
            tabfpn.fit(pfn_x, pfn_y)
            _, scores_train = tabfpn.predict(tmp_tmp_train_x, return_winning_probability=True)
            _, scores_test = tabfpn.predict(tmp_tmp_test_x, return_winning_probability=True)
            _, scores_val = tabfpn.predict(val_x, return_winning_probability=True)
            tmp_tmp_train_x[cat_list] = tmp_tmp_train_x[cat_list].astype("category")
            tmp_tmp_test_x[cat_list] = tmp_tmp_test_x[cat_list].astype("category")
            val_x[cat_list] = val_x[cat_list].astype("category")
            
            if stack:
                for i in range(scores_train.shape[1]):
                    tmp_tmp_train_x['scores_%d'%i] = scores_train[:,i]
                    tmp_tmp_test_x['scores_%d'%i] = scores_test[:,i]
                    val_x['scores_%d'%i] = scores_val[:,i]
        
            inner.append((tmp_tmp_train_x, tmp_tmp_test_x, val_x, tmp_tmp_train_y, tmp_test_y, val_y,
                        scores_train, scores_test, scores_val))
        
        output.append(inner)
    
    return output