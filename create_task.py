from Tablet import create
import os
import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="Run Optuna")
parser.add_argument("--name", default='Adult', type=str, help="data name")
args = parser.parse_args()

headers = {
    # "Abalone": "Given the parameters of an abalone, you must predict the age of the abalone from physical measurements. Sex is coded as as M, F, and I (infant).",
    # "Adult": "Given information about a person, you must predict if their income exceeds $50K/yr.",
    # "BreastCancer": "Given diagnostic information about a patient, you must predict if the patient has breast cancer.",
    # "Churn": "Given the following information about a customer, you must predict if the customer will churn.",
    # "HeartDisease": "Given the information about a patient in the heart disease database, you must predict if the patient has heart disease.",
    # "Sharktank": "Given the information of a pitch on the ABC show Shark Tank, you must predict if the pitch resulted in a deal (yes or no).",
    # "SoybeanLarge": "Given the following data, you must predict the Soybean disease.",
    # "Statlog": "Given the following data, you must predict whether to approve the credit card application.",
    # "UEFA": "Given the stats of a player, you must predict the number of goals the player scored in the champions league in a certain year.",
    # "Wine": "Given the following data, you must predict the origin of the wine.",
    # "bank": "Given the following data, you must predict if this client subscribes to a term deposit",
    # "blood": "Given the following data, you must predict if the person donated blood",
    # "calhousing": "Given the following data, you must predict if the house block is valuable",
    # "car": "Given the following data, you must rate the customers decision to buy this car",
    # "creditg": "Given the following data about a customer, you must predict whether their credit application was approved.",
    # "diabetes": "Given the diagnostic information about a patient, you must predict if the patient has diabetes.",
    # "heart": "Given the diagnostic information about a patient, you must predict if the patient has heart disease.",
    # "income": "Given the information about a person, you must predict if their income exceeds $50K/yr.",
    # "jungle": "Given the following data, you must predict if the white player wins this two pieces endgame of Jungle Chess",
    
    "balance-scale": "Given the left and right weights and distances, you must predict if the scale tips left or right, or is balanced.",
    "breast-w": "Given diagnostic information about a patient, you must predict if the patient has breast cancer.",
    "cmc": "Given the following demographic and socio-economic characteristics of a married woman, you must predict the current contraceptive method choice (no use, long-term methods, or short-term methods)",
    "credit-g": "Given the following data about a customer, you must predict whether their credit application was approved.",
    "diabetes": "Given the diagnostic information about a patient, you must predict if the patient has diabetes.",
    "tic-tac-toe": "Given the final board state of a tic-tac-toe game, you must predict if the game is won by the player with x.",
    "eucalyptus": "Given the following contributing factors such as height, diameter by height and survival, you must predict the soil retention utility of the specified eucalyptus seedlot.",
    "pc1": "Given the following data from flight software for earth orbiting satellite, you must predict if the software has defects.",
    "airlines": "Given the information of the scheduled departure, you must predict if the flight will be delayed.",
    "jungle_chess_2pcs_raw_endgame_complete": "Given the following data, you must predict if the white player wins this two pieces endgame of Jungle Chess",
    
}

class_map = {
    "balance-scale": {0: "left", 1: "balanced", 2: "right"},
    "breast-w": {0: "benign", 1: "malignant"},
    "cmc": {0: "no use", 1: "long-term methods", 2: "short-term methods"},
    "credit-g": {0: "no", 1: "yes"},
    "diabetes": {0: "no", 1: "yes"},
    "tic-tac-toe": {0: "no", 1: "yes"},
    "eucalyptus": {0: "low", 1: "low-medium", 2: "medium", 3: "medium-high", 4: "high"},
    "pc1": {0: "no", 1: "yes"},
    "airlines": {0: "no", 1: "yes"},
    "jungle_chess_2pcs_raw_endgame_complete": {0: "no", 1: "yes", 2: "draw"},
}

dataset_dir = "./data/" + args.name + "/prototypes-synthetic-performance-0"
train_x = pd.read_csv(dataset_dir + "/train.csv", index_col=0)
train_y = train_x['y_temp'].to_numpy()
train_x = train_x.drop(columns=['y_temp'])
eval_x = pd.read_csv(dataset_dir + "/test.csv", index_col=0)
eval_y = eval_x['y_temp'].to_numpy()
eval_x = eval_x.drop(columns=['y_temp'])

create.create_task(
    train_x,
    eval_x,
    train_y,
    eval_y,
    name=args.name, 
    header=headers[args.name],
    nl_instruction="",
    categorical_columns=["empty"],
    num=0,
    num_gpt3_revisions=10,
    openai_key_path="oai-key.txt",
    save_loc="./benchmark",
    only_natural_language=True
    )