from Tablet import evaluate

benchmark_path = "./data"
# benchmark_path = "./data/benchmark/performance"

import argparse
parser = argparse.ArgumentParser(description="Run Optuna")
parser.add_argument("--name", default='Adult', type=str, help="data name")
parser.add_argument("--model", default='google/flan-t5-xxl', type=str, help="model name")
parser.add_argument("--k_shot", default="3", type=int, help="k_shot")
parser.add_argument("--seed", default="0", type=int, help="seed")
args = parser.parse_args()

# tasks = [
#     args.name+'/prototypes-synthetic-performance-0',
#     ]

tasks = [
    args.name+'/prototypes-naturallanguage-performance-0',
    ]

if 'flan' in args.model:
    model_family = 'flan'
elif 'llama' in args.model or 'Qwen' in args.model:
    model_family = 'llama'
else:
    print("Invalid model selected for inference")

save_name = args.model.split('/')[-1]
    
save_paths = [
    args.name+'_{}_{}-shot_{}'.format(save_name, args.k_shot, args.seed),
    ]
evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                                tasks_to_run=tasks,
                                model=args.model,
                                encoding_format=model_family,
                                results_file=save_name+".txt",
                                k_shot=args.k_shot,
                                save_paths=save_paths)

evaluator.run_eval(how_many=1, seed=args.seed)
