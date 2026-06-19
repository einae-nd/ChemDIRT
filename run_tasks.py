from tensorboard import summary
import torch
import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

from utils.prompts import *
from utils.llm_utils import *
from utils.chem_utils import *
from utils.eval_utils import *

def main(model):

    model, tokenizer, generation_config = get_llm(args.model_name)
    print("Task Name:", args.task_name)
    print("Model:", args.model_name)
    print("Representation", args.format)
    print("Prompt Num:", args.rand)
    
    task = args.task_name

    df = pd.read_csv(f"data/all_questions_{args.format}_{args.rand}.csv")
        
    evaluation_fn = get_eval_type(task)

    task_df = df[df["task"] == task]

    q_batch = []
    a_batch = []
    img_batch = []
    all_preds = []
    
    task_prompt = get_task_prompt(task)
    
    for idx, row in task_df.iterrows():
        q_batch.append(row["question"])
        a_batch.append(row["answer"])
        
        if "image_path" in row and pd.notna(row["image_path"]):
            img_batch.append(row["image_path"])
        else:
            img_batch.append(None)
        
        if len(q_batch) == args.batch_size:
            preds = query_llm(q_batch, task_prompt, args.model_name, image_paths=img_batch, tokenizer=tokenizer, model=model, config=generation_config, max_tokens=args.max_tokens)

            for q, a, pred in zip(q_batch, a_batch, preds):
                all_preds.append({
                    "question": q,
                    "answer": a,
                    "prediction": pred
                })

            q_batch = []
            a_batch = []
            img_batch = []
            
    if len(q_batch) > 0:

        preds = query_llm(q_batch, task_prompt, args.model_name, image_paths=img_batch, tokenizer=tokenizer, model=model, config=generation_config, max_tokens=args.max_tokens)

        for q, a, pred in zip(q_batch, a_batch, preds):
            all_preds.append({
                "question": q,
                "answer": a,
                "prediction": pred
            })
    
    metric_values = defaultdict(list)

    for item in all_preds:
        results = evaluation_fn(
            item["prediction"],
            item["answer"]
        )

        for metric_name, value in results.items():
            if value is not None:
                metric_values[metric_name].append(value)
    
    summary = {
        metric: np.mean(values)
        for metric, values in metric_values.items()
    }
    
    print(f"Task: {task}")

    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")

    output_path = Path(f"outputs/{args.model_name}/{args.task_name}/results_{args.format}_{args.rand}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_preds, columns=["answer", "prediction", "question"]).to_csv(output_path, index=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='llama3', help='Name of the model to use for evaluation')
    argparser.add_argument('--max_tokens', type=int, default=128, help='Maximum number of tokens to generate')
    argparser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    argparser.add_argument('--rand', type=int, default=-1, help='Prompt selector')
    argparser.add_argument('--task_name', type=str, default='carbon_count', help='Type of question to create')
    argparser.add_argument('--format', type=str, default='both', choices=["smiles", "iupac", "both"], help='Representation format to use')
    args = argparser.parse_args()
    
    main(args)