import os
from autogluon.tabular import TabularPredictor
import pandas as pd
import argparse
import fcntl
import csv
import tempfile

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

def load_tabular_data(data_path, train_size=-1, seed=0):
    """Load tabular data."""
    
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")
    train_x = pd.read_csv(train_path, index_col=0)
    test_x = pd.read_csv(test_path, index_col=0)
    train_size = min(train_size, train_x.shape[0])
    
    if train_size > 0:
        train_x = train_x.sample(n=train_size, random_state=seed).reset_index(drop=True)
        
    return train_x, test_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna")
    parser.add_argument("--data_path", default='./data/adult_flan_t5/', type=str, help="Path to dataset")
    parser.add_argument("--train_size", default="-1", type=int, help="train_size")
    parser.add_argument("--seed", default="0", type=int, help="seed")
    args = parser.parse_args()
    
    train_data, test_data = load_tabular_data(args.data_path, train_size=args.train_size, seed=args.seed)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        predictor = TabularPredictor(label='y_temp', eval_metric='roc_auc_ovo_macro', path=tmpdir).fit(train_data)
        performance = predictor.evaluate(test_data)
        roc_auc_score = performance[str(predictor.eval_metric)]
        best_model_name = predictor.model_best
        
    # leaderboard = predictor.leaderboard(test_data)
    # print(leaderboard)

    # matching_row = leaderboard[leaderboard['model'] == best_model_name]

    # print(matching_row)

    # hyperparameters = {
    #     'GBM': {},  # Use default hyperparameters for Gradient Boosting Models
    #     'NN': {'num_epochs': 10},  # Train a neural network for 10 epochs
    #     'CAT': {'iterations': 1000}  # Use CatBoost with 1000 iterations
    # }

    # predictor = TabularPredictor(label='label').fit(train_data, hyperparameters=hyperparameters)
    print(performance)
    print(roc_auc_score)
    print(best_model_name)
    file_path = 'autogluon_compare.csv'
    data_to_append = [args.data_path.split("/")[-2], args.train_size, args.seed, best_model_name, roc_auc_score]

    append_line_to_csv(file_path, data_to_append)

