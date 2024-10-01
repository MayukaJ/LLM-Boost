"""Evaluate models."""
import logging
import os
# import json
from functools import partial
from typing import Any
# from sklearn.model_selection import train_test_split

import datasets
# from datasets import load_from_disk
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm
import torch
import transformers
import warnings
import xgboost
from xgboost import XGBClassifier
from catboost import Pool, CatBoostClassifier

from Tablet.prototypes import ProtoWrapper
from Tablet.ruleset import RSWrapper
from Tablet.synthetic_language import chatgpt_predict
from typing import Tuple

# Accepted ways of collating dataset components into prompts
ACCEPTED_ENCODING_FORMATS = [
    "tkinstruct",
    "tkinstruct-LIFT",
    "tabular",
    "flan",
    "flan-LIFT",
    "llama",
    "gpt",
    "gpt-LIFT"
]

# Custom specification for number of prototypes centroids
CENTROIDS = {
    "Breast Cancer Wisconsin (Diagnostic)": 2,
    "Heart Disease": 2,
    "Abalone": 3,
    "Breast Cancer Wisconsin (Original)": 2,
    "Thyroid Disease": 2,
    "Adult": 2,
    "churn": 2
}


# Cleans up quirks in text
def clean_up(text: str):
    """Remove formatting quirks"""
    return text.replace("with", "with ").replace("  ", " ").replace("\n ", "\n").replace("::", ":").replace(
        "The patient answered yes to the following questions "
        "about their symptoms.", "Here are the patient's "
                                 "responses to questions about "
                                 "their symptoms.").replace("\n\n\n", "\n\n")


def tk_instruct_few_shot_encoder(training_example: dict, k_shot: int):
    """Tk instruct few-shot collator.

    :param: training_example: training examples
    :param: k_shot: num k-shot examples
    """
    choices = np.random.choice(
        len(training_example['serialization']),
        size=k_shot,
        replace=False
    )
    lb = '\n'
    final = [
        f"Positive Example {j + 1} - Input: {training_example['serialization'][i]}"
        f"{(lb if not training_example['serialization'][i].endswith(lb) else '')}"
        f"Output: {training_example['label'][i]}" for j, i in enumerate(choices)
    ]
    final_k_shot_text = "\n".join(final)
    training_example["k_shot_samples"] = [final_k_shot_text] * len(training_example['serialization'])
    return training_example


def gpt_few_shot_encoder(training_example: dict, k_shot: int):
    """GPT few-shot collator.

    :param: training_example: training examples
    :param: k_shot: num k-shot examples
    """
    choices = np.random.choice(
        len(training_example['serialization']),
        size=k_shot,
        replace=False
    )
    lb = '\n'
    final = [
        f"\n\n{training_example['serialization'][i]}"
        f"{(lb if not training_example['serialization'][i].endswith(lb) else '')}"
        f"The answer is {training_example['label'][i]}" for j, i in enumerate(choices)
    ]
    final_k_shot_text = "\n".join(final)
    training_example["k_shot_samples"] = [final_k_shot_text] * len(training_example['serialization'])
    return training_example


def llama_few_shot_encoder(training_example: dict, k_shot: int):
    """GPT few-shot collator.

    :param: training_example: training examples
    :param: k_shot: num k-shot examples
    """
    choices = np.random.choice(
        len(training_example['serialization']),
        size=k_shot,
        replace=False
    )
    lb = '\n'
    final = [
        f"\n\n{training_example['serialization'][i]}"
        f"{(lb if not training_example['serialization'][i].endswith(lb) else '')}"
        f"The answer is {training_example['label'][i]}" for j, i in enumerate(choices)
    ]
    final_k_shot_text = "\n".join(final)
    training_example["k_shot_samples"] = [final_k_shot_text] * len(training_example['serialization'])
    return training_example


def flan_few_shot_encoder(training_example: dict, k_shot: int):
    """FLAN few-shot collator.

    :param: training_example: training examples
    :param: k_shot: num k-shot examples
    """
    choices = np.random.choice(
        len(training_example['serialization']),
        size=k_shot,
        replace=False
    )
    lb = '\n'
    final = [
        f"Example {j + 1} -\n{training_example['serialization'][i]}"
        f"{(lb if not training_example['serialization'][i].endswith(lb) else '')}"
        f"Answer: {training_example['label'][i]}" for j, i in enumerate(choices)
    ]
    final_k_shot_text = "\n".join(final)
    training_example["k_shot_samples"] = \
        [final_k_shot_text for _ in range(len(training_example['serialization']))]
    return training_example


def mod_data(data, nftu):
    """Modifies data to include only important features for selected few shot examples."""
    new_training_data = []
    for m in range(len(data["serialization"])):
        ser = data["serialization"][m]
        spser = ser.split("\n")
        cur = []
        for r in spser:
            for nf in nftu:
                if nf in r:
                    cur.append(r)
        if len(cur) == 1:
            cur += ["The patient did not have relevant symptoms."]
        new_training_data.append("\n".join(cur) + "\n")
    return new_training_data


def flan_few_shot_selection(test_example: Any,
                            training_data: dict,
                            orderings: np.ndarray,
                            k_shot: int = 4,
                            features_to_use: list[str] = None):
    """Selects KNN instances for prompt."""
    nftu = []
    for f in features_to_use:
        if " Response=" in f:
            nftu.append(f.split(" Response=")[0])
        else:
            nftu.append(f)
    nftu = set(nftu)
    nftu.add("Here are")
    new_training_data = mod_data(training_data, nftu)
    to_use_indices = orderings[:, :k_shot]
    k_shot_vals = []
    for k in range(len(test_example["serialization"])):
        choices = to_use_indices[k]
        choices = choices[::-1]  # put most similar first
        lb = '\n'
        final = [
            f"\n{new_training_data[i]}{(lb if not new_training_data[i].endswith(lb) else '')}"
            f"Answer: {training_data['label'][i]}\n" for j, i in enumerate(choices)
        ]
        final_k_shot_text = "\n".join(final).replace("::", ":")
        k_shot_vals.append(final_k_shot_text)
    return k_shot_vals, test_example


def process_tk_instruct(example: dict):
    """Collates dataset to prompts for tk-instruct + Instructions.

    Intended for use with dataset.map

    :param: example: the current example.
    """
    final = [
        (
            f"Definition: {example['header'][i]}\n{example['instructions'][i]} "
            f"{example['class_text'][i]}\n{example['k_shot_samples'][i]}\nNow complete "
            f"the following example - Input: {example['serialization'][i]}Output:"
        )
        for i in range(len(example['class_text']))]
    example["text"] = [clean_up(f) for f in final]
    return example


def process_tk_instruct_lift(example):
    """Collates dataset to prompts for tk-instruct + LIFT.

    Intended for use with dataset.map

    :param: example: the current example.
    """
    final = [
        (
            f"Definition: {example['lift_header'][i]}\n{example['class_text'][i]}\n"
            f"{example['k_shot_samples'][i]}\n"
            f"Now complete the following example - "
            f"Input: {example['serialization'][i]}Output:"
        )
        for i in range(len(example['class_text']))]
    example["text"] = [clean_up(f) for f in final]
    return example


def process_gpt_instruct(example):
    """Collates dataset to prompts for GPT + Instructions.

    Intended for use with dataset.map

    :param: example: the current example.
    """
    final = [
        (
            f"{example['header'][i]}\n{example['instructions'][i]} {example['class_text'][i]}\n"
            f"{example['k_shot_samples'][i]}\nNow complete "
            f"the following example - Input: {example['serialization'][i]}The answer is")
        for i in range(len(example['class_text']))]
    example["text"] = [clean_up(f) for f in final]
    return example

def process_llama_instruct(example):
    """Collates dataset to prompts for llama + Instructions.

    Intended for use with dataset.map

    :param: example: the current example.
    """
    final = [
        (
            f"{example['header'][i]}\n{example['instructions'][i]} Here are some examples.\n"
            f"{example['k_shot_samples'][i]}\n\nNow complete "
            f"the next example {example['class_text'][i].partition(' ')[2]} \n\n{example['serialization'][i]}. The final answer is: ")
        for i in range(len(example['class_text']))]
    example["text"] = [clean_up(f) for f in final]
    return example


def process_gpt_instruct_lift(example):
    """Collates dataset to prompts for GPT + LIFT.

    Intended for use with dataset.map

    :param: example: the current example.
    """
    final = [
        (
            f"{example['lift_header'][i]}\n{example['class_text'][i]}\n{example['k_shot_samples'][i]}"
            f"\nNow complete the following example - "
            f"Input: {example['serialization'][i]}The answer is")
        for i in range(len(example['class_text']))]
    example["text"] = [clean_up(f) for f in final]
    return example


def process_flan(example):
    """Collates dataset to prompts for Instructions + FLAN.

    Intended for use with dataset.map

    :param: example: the current example.
    """
    final = [
        (
            f"{example['header'][i]} {example['instructions'][i]} {example['class_text'][i]}"
            f"\n{example['k_shot_samples'][i]}\n\n"
            f"{example['serialization'][i]}Answer:")
        for i in range(len(example['class_text']))]
    example["text"] = [clean_up(f) for f in final]
    return example


def process_flan_lift(example):
    """Collates dataset to prompts for FLAN + LIFT.

    Intended for use with dataset.map

    :param: example: the current example.
    """
    final = [
        (f"{example['lift_header'][i]}"
         f"{example['class_text'][i]}\n{example['k_shot_samples'][i]}\n"
         f"{example['serialization'][i]}Answer:")
        for i in range(len(example['class_text']))]
    example["text"] = [clean_up(f) for f in final]
    return example


def chat_gpt_clean_up(output):
    response = "yes" if "the answer is yes" in output.lower() else "no"
    return response


class Evaluator:
    """Experiment evaluator.

    This class is used for data and evaluation management for a particular dataset. It supports things like
    running evaluation on a hugging face model or extracting a huggingface dataset object from the json files for
    a dataset, which you can include in your own evaluations.
    """

    def __init__(self,
                 benchmark_path: str,
                 encoding_format: str,
                 k_shot: int = 0,
                 results_file: str = "results.txt",
                 model: str = None,
                 tasks_to_run: list[str] = None,
                 save_paths: list[str] = None,
                 debug: bool = False,
                 use_tqdm: bool = True,
                 temp_y_column: str = "y_temp",
                 strategy: str = "random",
                 as_frac: bool = False,
                 how_many: int = 1,
                 seed: int = 0):
        """Init.

        :param benchmark_path: The path to the benchmark.
        :param encoding_format: The name of the encoding style to use. The options are provided in
                                ACCEPTED_ENCODING_FORMATS. This will dictate how the dataset components are collated
                                into a prompt.
        :param model: The name of the model. If encoding_format is 'tabular', the model name must be one supported,
                      e.g., L2LogReg for l2 regression. Otherwise, it is interpreted as the name of a huggingface
                      model. Set this to None if you're only interested in extracting the huggingface dataset.
        :param k_shot: The k-shot to evaluate with. If set > 0, will include k-shot examples extracted with :param:
                       strategy from the training dataset.
        :param: results_file: The file the results are appended to.
        :param: tasks_to_run: The paths of the tasks to run, given as their folder names. This folder should contain
                             the train.json and test.json. We'll validate this before attempting to run evaluation.
        :param: debug: If true, will run with a small subset of the testing data.
        :param: use_tqdm: If false, will remove tqdm calls.
        :param: temp_y_column: The name of the label column in the csvs.
        :param: strategy: The strategy for getting the k-shot examples. For true k-shot, leave to random. But, we
                          also support 'select' which will select the k closest instances in terms of euclidean
                          distance in the training set.
        :param: as_frac: If True, will interpret k_shot as a % of the total training set. E.g., k_shot=10 and as_frac=
                         True will result in 10% of the training data being included in the prompt.
        :param: how_many: How many times to run the evaluations. Will use different seeds for each eval.
        :param: seed: The seed to evaluate with.
        """
        self.how_many = how_many
        self.selection = None
        self.orderings = None
        self.as_frac = as_frac
        self.strategy = strategy
        self.encoding_format = encoding_format
        self.debug = debug
        self.tqdm = tqdm if use_tqdm else lambda x: x
        self.model = model
        self.results_file = results_file
        self.benchmark_path = benchmark_path
        self.tasks_to_run = tasks_to_run
        self.save_paths = save_paths
        self.accepted_encoding_formats = ACCEPTED_ENCODING_FORMATS
        self.temp_y_column = temp_y_column
        self.seed = seed
        self.k_shot = k_shot
        self.model_obj = None
        self.device = None
        self._validate()
        self._setup()

    def _validate(self):
        """Validates the evaluator."""
        if self.encoding_format not in self.accepted_encoding_formats:
            message = (f"Provided with encoding format {self.encoding_format}, but "
                       f"the accepted values are {self.accepted_encoding_formats}")
            raise ValueError(message)
        logging.info("Validated evaluator setup...")

    def _setup(self):
        """Sets things up."""
        if self.encoding_format == "tabular":
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        cache_dir = "/cmlscratch/user/hf"
        if self.model is None:
            self.tokenizer = None
            self.model_obj = None
        elif "EleutherAI" in self.model:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
            self.model_obj = transformers.AutoModelForCausalLM.from_pretrained(
                self.model
            ).to(self.device)
        elif "llama" in self.model:
            # self.tokenizer = transformers.LlamaTokenizer.from_pretrained(self.model)
            # self.model_obj = transformers.LlamaForCausalLM.from_pretrained(
            #     self.model, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16
            # )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model, 
                cache_dir=cache_dir
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # if "ul2" in self.model:
            self.model_obj = transformers.AutoModelForCausalLM.from_pretrained(
                self.model,
                load_in_8bit=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=cache_dir
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model, 
                cache_dir=cache_dir
                )
            # if "ul2" in self.model:
            self.model_obj = transformers.T5ForConditionalGeneration.from_pretrained(
                self.model,
                load_in_8bit=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=cache_dir
            )
            # else:
            #     self.model_obj = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            #         self.model, cache_dir=cache_dir
            #     ).to(self.device)

    def get_test_hf_datasets(self):
        """Gets the huggingface dataset objects for the tasks_to_run param provided at initialization."""
        return self.run_eval(how_many=1, seed=0, return_datasets=True)

    def run_eval(self, how_many: int, seed: int, return_datasets=False):
        """Runs evaluation.

        If return_datasets is True, just returns the hf test datasets.

        :param how_many: how many seeds to run on.
        :param return_datasets: if True, just returns hf datasets.
        """
        if self.tasks_to_run is None:
            # Should actually run all the tasks in the benchmark
            raise ValueError("No tasks specified to run.")
        else:
            hf_datasets = []
            for q in tqdm(range(how_many)):
                transformers.trainer_utils.set_seed(seed)
                for i, cur_t in enumerate(self.tqdm(self.tasks_to_run)):
                    task_path = os.path.join(self.benchmark_path, cur_t)
                    if self.is_valid_task(task_path):
                        if self.strategy == "select":
                            print(f"Processing closest inds for task {task_path}")
                            self.process_closest_inds(os.path.join(task_path, "train.csv"),
                                                      os.path.join(task_path, "test.csv"))
                        if self.encoding_format == "tabular":
                            results = self.evaluate_task_tabular(os.path.join(task_path, "train.csv"),
                                                                 os.path.join(task_path, "test.csv"))
                        elif self.encoding_format == "tkinstruct":
                            results = self.evaluate_task_nlp(os.path.join(task_path, "train.json"),
                                                             os.path.join(task_path, "test.json"),
                                                             os.path.join(task_path, "train.csv"),
                                                             os.path.join(task_path, "test.csv"),
                                                             encoding_f=process_tk_instruct,
                                                             k_shot_encoding_f=tk_instruct_few_shot_encoder,
                                                             return_datasets=return_datasets,
                                                             save_dir = self.save_paths[i])
                        elif self.encoding_format == "tkinstruct-LIFT":
                            results = self.evaluate_task_nlp(os.path.join(task_path, "train.json"),
                                                             os.path.join(task_path, "test.json"),
                                                             os.path.join(task_path, "train.csv"),
                                                             os.path.join(task_path, "test.csv"),
                                                             encoding_f=process_tk_instruct_lift,
                                                             k_shot_encoding_f=tk_instruct_few_shot_encoder,
                                                             return_datasets=return_datasets,
                                                             save_dir = self.save_paths[i])
                        elif self.encoding_format == "llama":
                            results = self.evaluate_task_nlp(os.path.join(task_path, "train.json"),
                                                             os.path.join(task_path, "test.json"),
                                                             os.path.join(task_path, "train.csv"),
                                                             os.path.join(task_path, "test.csv"),
                                                             encoding_f=process_llama_instruct,
                                                             k_shot_encoding_f=llama_few_shot_encoder,
                                                             return_datasets=return_datasets,
                                                             save_dir = self.save_paths[i])
                        elif self.encoding_format == "gpt":
                            results = self.evaluate_task_nlp(os.path.join(task_path, "train.json"),
                                                             os.path.join(task_path, "test.json"),
                                                             os.path.join(task_path, "train.csv"),
                                                             os.path.join(task_path, "test.csv"),
                                                             encoding_f=process_gpt_instruct,
                                                             k_shot_encoding_f=gpt_few_shot_encoder,
                                                             return_datasets=return_datasets,
                                                             save_dir = self.save_paths[i])
                        elif self.encoding_format == "gpt-LIFT":
                            results = self.evaluate_task_nlp(os.path.join(task_path, "train.json"),
                                                             os.path.join(task_path, "test.json"),
                                                             os.path.join(task_path, "train.csv"),
                                                             os.path.join(task_path, "test.csv"),
                                                             encoding_f=process_gpt_instruct_lift,
                                                             k_shot_encoding_f=gpt_few_shot_encoder,
                                                             return_datasets=return_datasets,
                                                             save_dir = self.save_paths[i])
                        elif self.encoding_format == "flan":
                            results = self.evaluate_task_nlp(os.path.join(task_path, "train.json"),
                                                             os.path.join(task_path, "test.json"),
                                                             os.path.join(task_path, "train.csv"),
                                                             os.path.join(task_path, "test.csv"),
                                                             encoding_f=process_flan,
                                                             k_shot_encoding_f=flan_few_shot_encoder,
                                                             return_datasets=return_datasets,
                                                             save_dir = self.save_paths[i])
                        elif self.encoding_format == "flan-LIFT":
                            results = self.evaluate_task_nlp(os.path.join(task_path, "train.json"),
                                                             os.path.join(task_path, "test.json"),
                                                             os.path.join(task_path, "train.csv"),
                                                             os.path.join(task_path, "test.csv"),
                                                             encoding_f=process_flan_lift,
                                                             k_shot_encoding_f=flan_few_shot_encoder,
                                                             return_datasets=return_datasets,
                                                             save_dir = self.save_paths[i])
                        # elif self.encoding_format == "tabllm":
                        #     results = self.evaluate_task_tabllm(os.path.join(task_path, "data"),
                        #                                      os.path.join(task_path, "train.csv"),
                        #                                      os.path.join(task_path, "test.csv"),
                        #                                      os.path.join(task_path, "note.json"),
                        #                                      save_dir = self.save_paths[i])
                        else:
                            raise ValueError(f"Bad encoding type {self.encoding_format}.")
                        if return_datasets:
                            hf_datasets.append(results)
                        else:
                            self.write_results(results, cur_t)
                    else:
                        raise ValueError(f"Invalid task {cur_t}!")
            return hf_datasets

    def write_results(self, results_dict: dict, task: str):
        """Writes final results."""
        results_save = {
            "task": [task],
            "model": [self.model],
            "encoding": [self.encoding_format],
            "references": [str(results_dict["references"])],
            "predictions": [str(results_dict["predictions"])],
            "f1": [results_dict["f1"]],
            "accuracy": [results_dict["accuracy"]],
            "k-shot": [self.k_shot]
        }
        results_save = pd.DataFrame(results_save)
        results_save.to_csv(self.results_file, mode="a", header=False)

    @staticmethod
    def is_valid_task(task_path: str):
        """Validates task."""
        # should_exist = ["train.json", "test.json", "train.csv", "test.csv"]
        # for s in should_exist:
        #     cur_task_path = os.path.join(task_path, s)
        #     if not os.path.exists(cur_task_path):
        #         raise ValueError(f"Invalid task {cur_task_path}!")
        return True

    def evaluate_task_nlp(self,
                          train_path: str,
                          test_path: str,
                          train_path_csv: str,
                          test_path_csv: str,
                          encoding_f: Any,
                          k_shot_encoding_f: callable,
                          return_datasets: bool = False,
                          save_dir = "test_dataset"):
        """Evaluates NLP task.

        :param train_path: The training data json path.
        :param test_path: The testing data json path.
        :param k_shot_encoding_f: The k-shot instance collating function.
        :param encoding_f: The prompt collating function.
        :param return_datasets: If true, will just return the hf dataset.
        """
        if self.debug:
            split = "test[:10]"
        else:
            split = "test"
        
        # split = "test[:10]" 
          
        test_data = self.get_test_dataset(
            encoding_f,
            k_shot_encoding_f,
            split,
            test_path,
            train_path
        )
        train_data = self.get_test_dataset(
            encoding_f,
            k_shot_encoding_f,
            "train",
            test_path,
            train_path
        )
        
        # Just return test data if specified
        
        references = test_data["label"]
        labels = np.unique(references)
        
        if return_datasets:
            return test_data
        if self.model == "chat-gpt" or self.model == "chat-gpt-trad":
            warning = "Currently, chatgpt prediction is implemented for the DDX datasets, and will lead to strange " \
                      "results on other dataset. It's recommended for you to write your own chatgpt eval, if not " \
                      "using these datasets."
            warnings.warn(warning)
            predictions = self.get_chat_gpt_predictions(test_data)
        else:
            predictions, test_scores = self.get_predictions(test_data, labels)
            train_predictions, train_scores = self.get_predictions(train_data, labels)
        
        f1 = f1_score(references,
                      predictions,
                      average="macro",
                      labels=labels)
        accuracy = accuracy_score(references, predictions)
        
        train_x, test_x, train_y, test_y = self.load_tabular_data(test_path_csv, train_path_csv)
        
        test_scores = np.transpose(np.array(test_scores))
        train_scores = np.transpose(np.array(train_scores))
        
        for i in range(test_scores.shape[0]):
            test_x["scores_%d"%i] = test_scores[i]
        for i in range(train_scores.shape[0]):
            train_x["scores_%d"%i] = train_scores[i]
        
        # test_x[self.temp_y_column] = test_y
        # train_x[self.temp_y_column] = train_y
        for i in range(len(labels)):
            test_x["labels_%d"%i] = np.where(test_y.astype(str) == labels[i], 1, 0)
        for i in range(len(labels)):
            train_x["labels_%d"%i] = np.where(train_y.astype(str) == labels[i], 1, 0)
            
        
        dataset_dir = os.path.join(self.benchmark_path, save_dir)
        os.makedirs(dataset_dir, exist_ok=True)
        
        test_x.to_csv(os.path.join(dataset_dir, "test.csv"))
        train_x.to_csv(os.path.join(dataset_dir, "train.csv"))
        
        eval_results = {
            "predictions": predictions,
            "references": references,
            "f1": f1,
            "accuracy": accuracy
        }
        return eval_results
    
    # def evaluate_task_tabllm(self,
    #                       data_path: str,
    #                       train_path: str,
    #                       test_path: str,
    #                       note_path: str,
    #                       save_dir = "test_dataset"):
        
    #     train_x = pd.read_csv(train_path, index_col=0)
    #     test_x = pd.read_csv(test_path, index_col=0)
    #     df = pd.concat([train_x, test_x], ignore_index=True)
    #     references = df[self.temp_y_column].to_list()
    #     y = df[self.temp_y_column].to_numpy()
    #     df = df.drop([self.temp_y_column], axis=1)
        
    #     labels = np.unique(references)
    #     with open(note_path, 'r') as file:
    #         data = json.load(file)
    #     extra_note = data["extra_note"]
        
    #     dataset = load_from_disk(data_path)
    #     predictions, scores = self.get_predictions(dataset, labels, data_label="note", extra_note=extra_note) 
        
    #     f1 = f1_score(references,
    #                   predictions,
    #                   average="macro",
    #                   labels=labels)
    #     accuracy = accuracy_score(references, predictions)
        
        
    #     scores = np.transpose(np.array(scores))
        
    #     for i in range(scores.shape[0]):
    #         df["scores_%d"%i] = scores[i]
        
    #     # test_x[self.temp_y_column] = test_y
    #     # train_x[self.temp_y_column] = train_y
    #     for i in range(len(labels)):
    #         df["labels_%d"%i] = np.where(y.astype(str) == labels[i], 1, 0)
        
    #     train_x, test_x, = train_test_split(df, test_size=0.20, shuffle=False)
          
    #     dataset_dir = os.path.join(self.benchmark_path, save_dir)
    #     os.makedirs(dataset_dir, exist_ok=True)
        
    #     test_x.to_csv(os.path.join(dataset_dir, "test.csv"))
    #     train_x.to_csv(os.path.join(dataset_dir, "train.csv"))
        
    #     eval_results = {
    #         "predictions": predictions,
    #         "references": references,
    #         "f1": f1,
    #         "accuracy": accuracy
    #     }
    #     return eval_results

    def get_test_dataset(self,
                         encoding_f: callable,
                         k_shot_encoding_f: callable,
                         split: str,
                         test_path: str,
                         train_path: str):
        """Generates the testing dataset.

        The testing dataset is stored in a huggingface dataset object.

        :param train_path: Path to the training json.
        :param test_path: Path to the testing json.
        :param split: The split to eval on.
        :param encoding_f: A function that collates the components in the data into a prompt, e.g., process_flan
        :param k_shot_encoding_f: A function that collates k_shot data.
        """
        test_data = datasets.load_dataset("json",
                                          split=split,
                                          data_files={
                                              "train": train_path,
                                              "test": test_path,
                                          })
        if self.k_shot > 0:
            train_data = datasets.load_dataset("json",
                                               split="train",
                                               data_files={
                                                   "train": train_path,
                                                   "test": test_path,
                                               })
            if self.strategy == "select":
                k_shot_data, test_data = flan_few_shot_selection(test_data,
                                                                 features_to_use=self.selection,
                                                                 training_data=train_data,
                                                                 k_shot=self.k_shot,
                                                                 orderings=self.orderings)
            else:
                train_data = train_data.map(partial(k_shot_encoding_f, k_shot=self.k_shot),
                                            batched=True,
                                            load_from_cache_file=False)
                k_shot_data = train_data["k_shot_samples"][0]
                k_shot_data = [k_shot_data for _ in range(len(test_data["serialization"]))]
        else:
            k_shot_data = ["" for _ in range(len(test_data["serialization"]))]
        test_data = test_data.add_column("k_shot_samples", k_shot_data)
        test_data = test_data.map(encoding_f, batched=True, load_from_cache_file=False)
        return test_data

    def get_chat_gpt_predictions(self, test_data):
        """Gets predictions from chat-gpt"""
        predictions, tokens = [], []
        for i in tqdm(range(len(test_data["text"]))):
            output = chatgpt_predict(test_data[i],
                                     lift="LIFT" in self.encoding_format,
                                     trad_prompt="-trad" in self.model)
            prediction, total_tokens = output
            prediction = chat_gpt_clean_up(prediction)
            predictions.append(prediction)
            tokens.append(total_tokens)
        tt = np.sum(tokens)
        cost = tt * 0.002 / 1_000
        print(f"Completions took {tt} tokens (${cost:.4f}")
        return predictions

    def get_predictions(self,
                        test_data, labels, data_label="text", extra_note=None):
        """Computes predictions on data."""
        all_texts = []
        scores = []
        
        if "llama" not in self.model:
            encoded_labels = []
            for label in labels:
                encoded_labels.append(self.tokenizer(label, 
                                                    return_tensors="pt").input_ids.to(self.device))
        
        for i in tqdm(range(len(test_data[data_label]))):
            if extra_note is not None:
                cur_text = test_data[data_label][i]+" "+extra_note
            else:
                cur_text = test_data[data_label][i]
            
            if "llama" in self.model:
                encoded_cur_labels = []
                for label in labels:
                    encoded_cur_labels.append(self.tokenizer(cur_text+" "+label, 
                                                     return_tensors="pt").input_ids.to(self.device))
            if "llama" in self.model:
                encoded_cur_text = self.tokenizer(cur_text+" ",
                                                return_tensors="pt").input_ids.to(self.device)
            else:
                encoded_cur_text = self.tokenizer(cur_text,
                                                return_tensors="pt").input_ids.to(self.device)
            start_index = encoded_cur_text.shape[-1]
            
            # Setup max lengths
            if "EleutherAI" in self.model:
                ml = len(encoded_cur_text[0]) + 5
            elif "llama" in self.model:
                ml = len(encoded_cur_text[0]) + 10 
            else:
                ml = 16
                
            # Fail if model is not initialized
            if self.model_obj is None:
                message = "Model is set to none... can't get predictions, terminating."
                raise ValueError(message)
            # Get generation
            # outputs = self.model_obj.generate(
            #     encoded_cur_text,
            #     do_sample=False,
            #     max_length=ml,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     repetition_penalty=1.1 if "llama" in self.model else 1.0,
            #     pad_token_id=self.tokenizer.pad_token_id if "llama" in self.model else None,
            # )
            
            logits = []
            
            if "llama" in self.model:
                for label in encoded_cur_labels:
                    masked_label = label.clone()
                    masked_label[0][:len(encoded_cur_text[0])] = torch.tensor(-100*np.ones(len(encoded_cur_text[0])))
                    loss = self.model_obj(input_ids=label, labels=masked_label).loss.item()
                    # loss = loss/(label.shape[-1] - encoded_cur_text.shape[-1] - 1)
                    logits.append(-loss)
            else:
                for label in encoded_labels:
                    loss = self.model_obj(input_ids=encoded_cur_text, labels=label).loss.item() #.logit
                    # loss = loss/(label.shape[-1] - 1)
                    logits.append(-loss)
            
            # logits = [np.exp(prob) for prob in logits]
            # logits = [(logit+1e-8)/sum(logits) for logit in logits]
            # logits = [np.log(prob) for prob in logits]
            logits = [logit-min(logits) for logit in logits]
            
            
            # transition_scores = self.model_obj.compute_transition_scores(
            #     outputs.sequences, outputs.scores, normalize_logits=True
            # )
            # final_score = np.exp(np.sum(transition_scores[0].cpu().numpy()))
            
            # generation = outputs.sequences
            # # Push generation back locally
            # generation = generation.cpu().numpy().tolist()

            # Decode
            # if "EleutherAI" in self.model or "llama" in self.model:
            #     text_generation = self.tokenizer.decode(
            #         generation[0][start_index:],
            #         #generation[0][len(encoded_cur_text[0]):],
            #         skip_special_tokens=True
            #     )
            #     text_generation = text_generation.split(".")[0]
            # else:
            #     text_generation = self.tokenizer.decode(
            #         generation[0],
            #         skip_special_tokens=True
            #     )
            #     text_generation = text_generation.rstrip(".")
            # all_texts.append(text_generation)
            all_texts.append("None")
            scores.append(logits)
        return all_texts, scores

    def evaluate_task_tabular(self, train_path: str, test_path: str):
        """Evaluates fully supervised models.

        :param train_path: Training csv path.
        :param test_path: Testing csv path.
        """
        tab_data = self.load_tabular_data(
            test_path,
            train_path
        )
        train_x, test_x, train_y, test_y = tab_data
        cat_features = [0,2,4,6,8,9]
        eval_model, enc_data = self._get_tabular_model(train_path)
        # One hot encode categorical features
        
        if self.model != "Catboost":
            len_train_x = len(train_x)
            all_data = pd.concat([train_x, test_x], axis=0)
            all_data_dummies = pd.get_dummies(all_data)
            train_x = all_data_dummies.iloc[:len(train_x)]
            test_x = all_data_dummies.iloc[len(train_x):]
            assert len(train_x) == len_train_x
            
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        if self.as_frac:
            k_shot_pct = self.k_shot / 100.0
            assert 1.0 >= k_shot_pct >= 0, "k_shot/100 must be on range 0 -> 1"
            to_use = np.random.choice(len(train_x), size=int(len(train_x) * k_shot_pct), replace=False)
            train_x = train_x.iloc[to_use]
            train_y = train_y[to_use]
        elif self.k_shot > 0:
            to_use = np.random.choice(len(train_x), size=self.k_shot, replace=False)
            train_x = train_x.iloc[to_use]
            train_y = train_y[to_use]
            
        
        ss = StandardScaler()
        if enc_data:
            train_x = ss.fit_transform(train_x)
            test_x = ss.transform(test_x)
        test_y = le.transform(test_y)
        # Get model
        # test_y = np.ravel(test_y)
        # train_y = np.ravel(train_y)
        if len(np.unique(train_y)) == 1:
            predictions = [np.unique(train_y)[0] for _ in range(len(test_x))]
        else:
            if self.model != "Catboost":
                eval_model.fit(train_x, train_y)
                predictions = eval_model.predict(test_x)
            else:
                # train_dataset = Pool(data=train_x,
                #      label=train_y,
                #      cat_features=cat_features)
                # eval_dataset = Pool(data=test_x,
                #     label=test_y,
                #     cat_features=cat_features)
                # eval_model.fit(train_dataset)
                # predictions = eval_model.predict(eval_dataset)
                eval_model.fit(train_x, train_y, cat_features=cat_features)
                print(test_x)
                predictions = eval_model.predict(test_x)
                
        semantic_predictions = le.inverse_transform(predictions)
        f1s = f1_score(test_y, predictions, average="macro", labels=np.unique(test_y))
        accuracy = accuracy_score(test_y, predictions)
        results = {
            "predictions": semantic_predictions.tolist(),
            "references": le.inverse_transform(test_y).tolist(),
            "f1": f1s,
            "accuracy": accuracy
        }
        return results

    def load_tabular_data(self, test_path, train_path):
        """Load tabular data."""
        train_x = pd.read_csv(train_path, index_col=0)
        train_y = train_x[self.temp_y_column].to_numpy()
        test_x = pd.read_csv(test_path, index_col=0)
        test_y = test_x[self.temp_y_column].to_numpy()
        test_x = test_x.drop([self.temp_y_column], axis=1)
        train_x = train_x.drop([self.temp_y_column], axis=1)
        return train_x, test_x, train_y, test_y

    def _get_tabular_model(self, path: str):
        """Setup tabular models.

        We just support a fixed set of tabular models, which are loaded by their names. Will raise a ValueError
        if the model is unknown.

        Current supported options are:
            - L2 Regression: L2LogReg
            - XGBoost: XGB
            - Linear SVM: SVC
            - Rule Set: RS
            - Prototypes: prototypes
        """
        if self.model == "L2LogReg":
            eval_model = LogisticRegression()
            enc_data = True
        elif self.model == "XGB":
            eval_model = xgboost.XGBClassifier()
            enc_data = False
        elif self.model == "Catboost":
            eval_model = CatBoostClassifier(
                                            # loss_function=LoglossObjective(),
                                            # eval_metric=LoglossMetric(),
                                            iterations=10,
                                            loss_function='MultiClass', 
                                            # task_type="GPU", devices='0'
                                            )
            enc_data = False
        elif self.model == "SVM":
            eval_model = SVC()
            enc_data = True
        elif self.model == "RS":
            enc_data = False
            eval_model = RSWrapper(max_depth=1)
        elif self.model == "prototypes":
            enc_data = False  # note that flipping this to true will cause error, due to prototypes imp.
            split_path = path.split("/")
            # allow custom centroids configurations
            if len(split_path) >= 3 and split_path[2] in CENTROIDS:
                print(f"Using custom centroids num {CENTROIDS[split_path[2]]}.")
                eval_model = ProtoWrapper(n_clusters=CENTROIDS[split_path[2]])
            else:
                eval_model = ProtoWrapper(n_clusters=2)
        elif self.model == "rulesets":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown tabular model {self.model}.")
        return eval_model, enc_data

    def process_closest_inds(self, train_path, test_path):
        """Pre-processes self.k_shot number closest training points to each test point."""
        train_x_df, test_x, train_y, test_y = self.load_tabular_data(test_path, train_path)
        cols = np.array(train_x_df.columns.tolist())
        # One-hot encode
        all_data = pd.concat([train_x_df, test_x], axis=0, ignore_index=True)
        all_data_np = pd.get_dummies(all_data).to_numpy()
        train_x, test_x = all_data_np[:len(train_x_df)], all_data_np[len(train_x_df):]
        # label encode
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        # annoying xgb setup, fails otherwise
        train_x_df[train_x_df.select_dtypes(['object']).columns] = train_x_df.select_dtypes(
            ['object']).apply(lambda x: x.astype('category'))
        train_x_df[train_x_df.select_dtypes(['bool']).columns] = train_x_df.select_dtypes(
            ['bool']).apply(lambda x: x.astype('category'))
        # train xgb
        gbm = XGBClassifier(tree_method="hist",
                            enable_categorical=True).fit(train_x_df, train_y)
        # compute distances
        distances = np.sqrt(np.sum(((test_x[:, None, :] - train_x) ** 2) * 1, axis=2))
        orderings = np.argsort(distances, axis=1)
        # also select most important features, these are used to present only most important features
        # in the prompts
        selection = SelectFromModel(gbm, threshold=0.001, prefit=True).get_support()
        self.selection = cols[selection]
        self.orderings = orderings
