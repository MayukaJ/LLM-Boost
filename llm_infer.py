from Tablet import evaluate

benchmark_path = "./data"

import argparse
parser = argparse.ArgumentParser(description="Run Optuna")
parser.add_argument("--name", default='Adult', type=str, help="data name")
parser.add_argument("--model", default='flan', type=str, help="data name")
parser.add_argument("--seed", default="0", type=int, help="seed")
args = parser.parse_args()

tasks = [
    args.name+'/prototypes-synthetic-performance-0',
    ]

if args.model == 'flan':
    save_paths = [
        args.name+'_flan',
        ]
    evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                                    tasks_to_run=tasks,
                                    model='google/flan-t5-xxl',
                                    encoding_format="flan",
                                    results_file="results_2.txt",
                                    k_shot=3,
                                    save_paths=save_paths)

elif args.model == 'llama':
    save_paths = [
        args.name+'_llama',
        ]
    evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                                    tasks_to_run=tasks,
                                    model='meta-llama/Meta-Llama-3-8B-Instruct',
                                    encoding_format="llama",
                                    results_file="llama3.txt",
                                    k_shot=3,
                                    save_paths=save_paths)

else:
    print("Invalid model selected for inference")
    assert False

evaluator.run_eval(how_many=1, seed=args.seed)
