# %% imports and settings
import json
import os
from functools import wraps
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from typing import Dict, List, Tuple
import argparse
from IPython import get_ipython

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model-path",
    type=str,
    required=True,
    help="Path to the model to evaluate",
)
parser.add_argument(
    "-d",
    "--data-path",
    type=str,
    required=True,
    help="Path to the data to evaluate",
)
parser.add_argument(
    "-l",
    "--max-new-tokens",
    type=int,
    required=False,
    default=256,
    help="Max number of new tokens to generate",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=64,
    required=False,
    help="Batch size for generation",
)
# TODO: rename the dir
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=False,
    default="/mnt/default/eval_results/",
)
parser.add_argument(
    "--replace",
    action="store_true",
    default=False,
    help="Replace the saved file with the generated file",
)


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if isnotebook():
    args = parser.parse_args(
        args=[
            "--model-path",
            "/home/v-haotiancui/NL2Code/Copilot-2/code2text/codeT5/results/ft-stage2-codeT5-GithubCodeDocStep1n2Filtered",
            "--data-path",
            "/home/v-haotiancui/NL2Code/Copilot-2/dataset/GithubCodeDocStep1n2Filtered.holdout.jsonl",
            "--batch-size",
            "64",
            "--save-dir",
            "/home/v-haotiancui/NL2Code/Copilot-2/code2text/codeT5/results/",
            "--replace",
        ]
    )
else:
    args = parser.parse_args()

"""
for local test
python eval-gpt-neo.py \
    --model-path /home/v-haotiancui/NL2Code/Copilot-2/code2text/codeT5/results/ft-stage2-codeT5-GithubCodeDocStep1n2Filtered \
    --data-path /home/v-haotiancui/NL2Code/Copilot-2/dataset/GithubCodeDocStep1n2Filtered.holdout.jsonl \
    --batch-size 64 \
    --save-dir /home/v-haotiancui/NL2Code/Copilot-2/code2text/codeT5/results/ \
    --replace

for amlk8s
python eval-codeT5.py \
    --model-path /mnt/default/ft-stage2-codeT5-GithubCodeDocStep1n2Filtered-Mar12-20-28-2022/checkpoint-14000
    --data-path /mnt/default/data/GithubCodeDocStep1n2Filtered.holdout.jsonl 
    --save-dir /mnt/default/eval_results/holdout_test/
    --max-new-token 512
    --batch-size 256

"""

# print args
print(args)

output_file_jsonl = os.path.join(
    args.save_dir,
    "eval-"
    + args.model_path.split("/")[-2]
    + "-"
    + args.model_path.split("/")[-1]
    + ".jsonl",
)
output_file_csv = os.path.join(
    args.save_dir,
    "eval-"
    + args.model_path.split("/")[-2]
    + "-"
    + args.model_path.split("/")[-1]
    + ".csv",
)
# mkdir if needed
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if os.path.exists(output_file_jsonl):
    if args.replace:
        print(
            f"Result already exists. Going to replace"
            f" {output_file_jsonl} and {output_file_csv}"
        )
    else:
        print(f"File {output_file_jsonl} already exists, skipping evaluation")
        exit()
print(f"Evaluating model at {args.model_path}")
print(f"Save to {output_file_jsonl}, {output_file_csv}")
# %% load the model
def load_model(model_path: str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load a model from a path.

    Args:
        model_path: Path to the model.

    Returns:
        model: The model.
        tokenizer: The tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    # put model to gpu if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# read jsonl file
def load_code_doc(data_file: str):
    assert os.path.exists(data_file)
    if data_file.endswith(".jsonl"):
        with open(data_file) as f:
            data = [json.loads(line) for line in f]
        if "docstring" in data[0]:
            docs = [d["docstring"] for d in data]
        else:
            docs = [d["doc"] for d in data]
        block_codes = (
            [d["blocks_codes"] for d in data] if "blocks_codes" in data[0] else None
        )
        codes = [d["code"] for d in data]
    elif data_file.endswith(".json"):
        with open(data_file) as f:
            data = json.load(f)
        docs = [d["docstring"] for d in data]
        block_codes = None
        codes = [d["code"] for d in data]
    return docs, block_codes, codes


# %% generate function


def generate(source_texts, tokenizer, model, max_new_tokens=2048):
    """Generate text using a model. Requires encoder-decoder model, e.g. T5."""
    if max_new_tokens > tokenizer.model_max_length:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) should be smaller than"
            f" model max length ({tokenizer.model_max_length})"
        )
    input = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)
    input = input.to(model.device)

    print(f"Generating {max_new_tokens} tokens")
    gen_tokens = model.generate(
        **input,
        do_sample=True,
        temperature=0.1,
        max_length=max_new_tokens,
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text


# %% actual run
model, tokenizer = load_model(args.model_path)
docs, _, codes = load_code_doc(args.data_path)
batch_size = args.batch_size
max_new_tokens = args.max_new_tokens
res = []
for i in range(0, len(codes), batch_size):
    print(f"running example {i}/{len(codes)}")
    batch = codes[i : i + batch_size]
    gen_text = generate(batch, tokenizer, model, max_new_tokens=max_new_tokens)
    for j, text in enumerate(gen_text):
        d = {
            "gen": text,
            "source": batch[j],
            "reference": docs[i + j],
        }
        res.append(d)

if args.replace or not os.path.exists(output_file_jsonl):
    with open(output_file_jsonl, "w") as f:
        for d in res:
            f.write(json.dumps(d) + "\n")

# also make a csv
if args.replace or not os.path.exists(output_file_csv):
    df = pd.DataFrame(res)
    df.to_csv(output_file_csv, index=True)

# %%
