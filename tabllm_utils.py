from pathlib import Path
import datasets
import numpy as np
import pandas as pd
# import xgboost as xgboost
from datasets import DatasetDict, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split

datasets.enable_caching()


def load_tabllm_data(data_dir, seed, num_shot, cv_folds=5):
    # num_shots = [4, 8, 16, 32, 64, 128, 256, 512, 'all']  # , 1024, 2048, 4096, 8192, 16384, 50000, 'all']  # ['all']
    # seeds = [42, 1024, 0, 1, 32]   # , 45, 655, 186, 126, 836]
    categorical_encoding = 'one-hot'

    data_dir = Path(data_dir)
    train_dataset = pd.read_csv(data_dir / 'train.csv', index_col=[0])
    test_dataset = pd.read_csv(data_dir / 'test.csv', index_col=[0])
    n_classes = len([col for col in train_dataset.columns if 'labels' in col])
    dataset = {'train': train_dataset, 'test': test_dataset}
    
    print(f"Original columns: {list(dataset['train'].columns)}")
    dataset = DatasetDict({k: Dataset.from_pandas(v, preserve_index=False) for k, v in dataset.items()})
    dataset = concatenate_datasets(list(dataset.values()))
    
    dataset = DatasetDict({k: read_orig_dataset(dataset, seed, k) for k in ['train', 'test']})

    dataset = DatasetDict({k: v.to_pandas() for k, v in dataset.items()})
    dataset = prepare_data(dataset, enc=categorical_encoding, scale=False)
    print(f"prepared columns: {list(dataset['train'].columns)}")

    # Load into hf datasets to replicate tfew methods
    dataset = {k: Dataset.from_pandas(v, preserve_index=False) for k, v in dataset.items()}
    dataset = DatasetDict(dataset)

    # dataset_validation = dataset['validation'].remove_columns(['idx'])
    dataset_test = dataset['test'].remove_columns(['idx'])

    if num_shot == 'all':
        dataset_train = (dataset['train'].remove_columns(['idx'])).shuffle(seed)
        # dataset_validation = dataset_validation.shuffle(seed)
        dataset_test = dataset_test.shuffle(seed)
    else:
        dataset_train = sample_few_shot_data(dataset['train'], num_shot, seed)
        dataset_train = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset_train))
        # print(f"Final columns: {list(dataset_train.columns)}")
        dataset_train = dataset_train.remove_columns(['idx'])
        
        X_train = dataset_train.to_pandas()
        y_train = pd.DataFrame()
        for i in range(n_classes):
            y_train['labels_%d'%i] = X_train['labels_%d'%i]
            X_train = X_train.drop(['labels_%d'%i], axis=1)
        y_train = np.array(y_train).astype('float32')
        scores_train = pd.DataFrame()
        for i in range(n_classes):
            scores_train['scores_%d'%i] =  X_train['scores_%d'%i]
            X_train =  X_train.drop(['scores_%d'%i], axis=1)
        scores_train = np.array(scores_train).astype('float32')
        scores_train = scores_train - np.max(scores_train, axis=1, keepdims=True)/2
        
        X_test = dataset_test.to_pandas()
        y_test = pd.DataFrame()
        for i in range(n_classes):
            y_test['labels_%d'%i] = X_test['labels_%d'%i]
            X_test = X_test.drop(['labels_%d'%i], axis=1)
        y_test = np.array(y_test).astype('float32')
        scores_test = pd.DataFrame()
        for i in range(n_classes):
            scores_test['scores_%d'%i] =  X_test['scores_%d'%i]
            X_test =  X_test.drop(['scores_%d'%i], axis=1)
        scores_test = np.array(scores_test).astype('float32')
        scores_test = scores_test - np.max(scores_test, axis=1, keepdims=True)/2
        
    output = []
    for val_iter in range(cv_folds):
        tmp_train_x, val_x, tmp_train_y, val_y, tmp_scores_train, scores_val = train_test_split(X_train, y_train, scores_train,
                                                                                            test_size=0.5, random_state=val_iter+seed, stratify=y_train)
        
        output.append((tmp_train_x, X_test, val_x, tmp_train_y, y_test, val_y,
                        tmp_scores_train, scores_test, scores_val))
    
    
    return [output]


# def evaluate_model(seed, model, metric, parameters, X_train, y_train, X_valid, y_valid, X_test, y_test):
#     print(f"\tUse {X_train.shape[0]} train, {X_valid.shape[0]} valid, and {X_test.shape[0]} test examples.")

#     def get_lr():
#         # Kept balanced bc for all 'shot' experiments no effect, only used for 'all' which should be better.
#         # In contrast to IBC: removed tol=1e-1
#         return LogisticRegression(class_weight='balanced', penalty='l1', fit_intercept=True, solver='liblinear',
#                                   random_state=seed, verbose=0, max_iter=200)

#     def get_light_gbm():
#         return lgb.LGBMClassifier(class_weight='balanced', num_threads=1, random_state=seed)

#     def get_xgboost():
#         # No class_weight parameter, only scale_pos_weight, but should not be a difference for all shot experiments.
#         # eval_metric gives same as non, but without a warning
#         return xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss', nthread=1, random_state=seed)

#     def get_tabpfn():
#         # Use default configuration
#         return TabPFNClassifier()

#     def compute_metric(clf_in, X, y):
#         if metric == 'roc_auc':
#             p = clf_in.predict_proba(X)[:, 1]
#             metric_score = roc_auc_score(y, p)
#         elif metric == 'roc_auc_ovr':
#             p = clf_in.predict_proba(X)
#             metric_score = roc_auc_score(y, p, multi_class='ovr', average='macro')
#         elif metric == 'accuracy':
#             p = np.argmax(clf_in.predict_proba(X), axis=1)
#             metric_score = np.sum(p == np.array(y)) / p.shape[0]
#         else:
#             raise ValueError("Undefined metric.")
#         return metric_score

#     # Do a 4-fold cross validation on the training set for parameter tuning
#     folds = min(Counter(y_train).values()) if min(Counter(y_train).values()) < 4 else 4  # If less than 4 examples
#     if folds < 4:
#         print(f"Manually reduced folds to {folds} since this is maximum number of labeled examples.")

#     if folds > 1:
#         inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
#     else:
#         print(f"Warning: Increased folds from {folds} to 2 (even though not enough labels) and use simple KFold.")
#         folds = 2
#         inner_cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
#     estimator = None
#     if model == 'lr':
#         estimator = get_lr()
#     elif model == 'lightgbm':
#         estimator = get_light_gbm()
#     elif model == 'xgboost':
#         estimator = get_xgboost()
#     elif model == 'tabpfn':
#         estimator = get_tabpfn()
#     # Add verbose 4 for more detailed output
#     clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv, scoring=metric, n_jobs=40, verbose=0)
#     clf.fit(X_train, y_train)

#     # In depth debug linear model to determine support of each patient
#     # if model == 'lr':
#     #     print(len(list(est.coef_[0])))
#     #     print(sum([(x != 0) for x in list(est.coef_[0])]))
#     #     print(clf.best_params_)
#     #     est = clf.best_estimator_
#     #     print('Coefficients:', len(list(est.coef_[0])), 'non-zero:', sum([(x != 0) for x in list(est.coef_[0])]))
#     #     print('Coefficients wout static:', len(list(est.coef_[0])) - 9, '/3 = ', (len(list(est.coef_[0])) - 9) / 3)
#     #     # Determine non-zero weights in test set
#     #     weights_per_person = X_test.toarray() * est.coef_[0]
#     #     # Remove nine trailing static features
#     #     weights_per_person = weights_per_person[:, :-9]
#     #     non_zero = np.count_nonzero(weights_per_person, axis=1)
#     #     print('Windowed', np.median(non_zero), np.quantile(non_zero, 0.25), np.quantile(non_zero, 0.75), np.max(non_zero))
#     #     # Non zero for concept, so summed up windows
#     #     weights_per_person = weights_per_person.reshape((weights_per_person.shape[0], 3, int(weights_per_person.shape[1]/3)))
#     #     weights_per_person = np.sum(weights_per_person, axis=1)
#     #     non_zero = np.count_nonzero(weights_per_person, axis=1)
#     #     print('Not windowed', np.median(non_zero), np.quantile(non_zero, 0.25), np.quantile(non_zero, 0.75), np.max(non_zero))

#     score_test = compute_metric(clf, X_test, y_test)
#     return score_test

#     #     one_hot_test = pd.get_dummies(dataset['test'][[c for c in list(dataset['test']) if c != 'label']])[columns]
#     #     pred = np.argmax(lr.predict_proba(one_hot_test), axis=1)
#     #     acc_test = np.sum(pred == np.array(dataset['test']['label'].tolist()))/pred.shape[0]
#     #     print("C: %.4f, Val acc: %.2f, Test acc: %.2f" % (C, acc_valid, acc_test))
#     # return


def prepare_data(dataset, enc=None, scale=True):
    def create_valid_test(split):
        # Replicate steps performed on training data
        data = dataset[split]

        # Add all columns that are in test but not here
        for column in [c for c in dataset['train'].columns if c not in data.columns]:
            data[column] = 0.
        # Put columns in same order as test and remove everything that is not in test
        return data[dataset['train'].columns]

    # dataset['validation'] = create_valid_test('validation')
    dataset['test'] = create_valid_test('test')
    assert (len(dataset['train'].columns) == len(dataset['test'].columns))
    return dataset


def read_orig_dataset(orig_data, seed, split):
    # External datasets are not yet shuffled, so do it now
    data = orig_data.train_test_split(test_size=0.20, seed=seed)
    data2 = data['test'].train_test_split(test_size=0.50, seed=seed)
    # No validation/test split used for external datasets
    dataset_dict = DatasetDict({'train': data['train'],
                                'test': concatenate_datasets([data2['train'], data2['test']]),
                                'validation': Dataset.from_dict({'note': [], 'label': []})})
    orig_data = dataset_dict[split]

    # In case dataset has no idx per example, add that here bc manually created ones might not have an idx.
    if 'idx' not in orig_data.column_names:
        orig_data = orig_data.add_column(name='idx', column=range(0, orig_data.num_rows))

    return orig_data


def sample_few_shot_data(orig_data, num_shot, few_shot_random_seed):
    orig_data = convert_one_hot_to_single_column(orig_data)
    saved_random_state = np.random.get_state()
    np.random.seed(few_shot_random_seed)
    # Create a balanced dataset for categorical data
    labels = {label: len([ex['idx'] for ex in orig_data if ex['label'] == label])
              for label in list(set(ex['label'] for ex in orig_data))}
    num_labels = len(labels.keys())
    ex_label = int(num_shot / num_labels)
    ex_last_label = num_shot - ((num_labels - 1) * ex_label)
    ex_per_label = (num_labels - 1) * [ex_label] + [ex_last_label]
    assert sum(ex_per_label) == num_shot

    # Select num instances per label
    old_num_labels = []
    datasets_per_label = []
    for i, label in enumerate(labels.keys()):
        indices = [ex['idx'] for ex in orig_data if ex['label'] == label]
        old_num_labels.append(len(indices))
        # Sample with replacement from label indices
        samples_indices = list(np.random.choice(indices, ex_per_label[i]))
        datasets_per_label.append(orig_data.select(samples_indices))
    orig_data = concatenate_datasets(datasets_per_label)

    # Check new labels
    old_labels = labels
    labels = {label: len([ex['idx'] for ex in orig_data if ex['label'] == label])
              for label in list(set(ex['label'] for ex in orig_data))}
    print(f"Via sampling with replacement old label distribution {old_labels} to new {labels}")
    assert sum(labels.values()) == num_shot
    assert len(orig_data) == num_shot
    
    orig_data = orig_data.remove_columns(['label'])

    np.random.set_state(saved_random_state)
    # Now randomize and (selection of num_shots redundant now bc already done).
    # Call to super method directly inserted here
    saved_random_state = np.random.get_state()
    np.random.seed(few_shot_random_seed)
    orig_data = [x for x in orig_data]
    np.random.shuffle(orig_data)
    selected_data = orig_data[: num_shot]
    np.random.set_state(saved_random_state)
    return selected_data


def convert_one_hot_to_single_column(dataset, prefix='labels_'):
    # Find all columns that start with the given prefix
    one_hot_columns = [col for col in dataset.column_names if col.startswith(prefix)]
    
    # Create a function to map one-hot encoded rows to a single value
    def one_hot_to_label(row):
        for col in one_hot_columns:
            if row[col] == 1:
                return col[len(prefix):]
        return None
    
    # Apply the function to each row
    labels = dataset.map(lambda row: {'label': one_hot_to_label(row)}, batched=False)
    
    # Remove the one-hot columns
    # dataset = dataset.remove_columns(one_hot_columns)
    
    # Add the new 'label' column
    dataset = dataset.add_column('label', labels['label'])
    
    return dataset

