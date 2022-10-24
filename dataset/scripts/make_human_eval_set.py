# %% [markdown]
# # Make Human Eval Set
# This script:
# 1. merge result files from different models into one file.
# 2. select 180 samples and collect codex results
# 4. save all results and original index in the merged file for future reference.
# 3. prepare files for external annotators, and store the mapping file (internal key)

# %%
import json
import os
import time
import random
import pandas as pd

from typing import Dict, List, Literal, TextIO, Tuple
import argparse
from IPython import get_ipython
from utils import load_code_doc, load_key_jsonl, isnotebook, call_codex

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--source-file",
    type=str,
    required=False,
    default=(
        "/home/v-haotiancui/NL2Code/Copilot-2/dataset/"
        "GithubCodeDocStep1n2Filtered.test.jsonl"
    ),
    help="Path to the file containing the source code and reference doc to evaluate",
)
parser.add_argument(
    "-d",
    "--data-dir",
    type=str,
    required=False,
    default="/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_3k_testset/",
    help="Path to the data directory to evaluate. This is the derictory containing"
    " source code and generated docs from various models. Samples will be selected"
    " from these pairs.",
)
parser.add_argument(
    "-m",
    "--models",
    type=str,
    required=False,
    default=None,
    help="list of model names to add into the human-eval set, separated by comma.",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=False,
    default="/home/v-haotiancui/NL2Code/Copilot-2/dataset/human-eval/",
    help="Directory to save the human-eval set",
)
parser.add_argument(
    "-n",
    "--num-samples",
    type=int,
    required=False,
    default=180,
    help="Number of samples to select for the human-eval set",
)
parser.add_argument(
    "--replace",
    action="store_true",
    default=False,
    help="Replace the saved file with the generated file",
)
parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=42,
    help="Random seed for sampling",
)

if isnotebook():
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()
source_file = args.source_file
data_dir = args.data_dir
models = (
    [
        "ft12_gpt2",
        "ft12_gpt-neo",
        "ft12_gpt-neo-27",
        "ft1_codeT5",
        "ft2_codeT5",
        "ft12_codeT5",
    ]
    if args.models is None
    else [m.strip() for m in args.models.split(",")]
)
save_dir = args.save_dir
num_samples = args.num_samples

random.seed(args.seed)

# check per model file exists
data_files = {
    "ft12_gpt2": "eval-ft-stage2-gpt2-GithubCodeDocStep1n2Filtered-Jan21-13-24-2022-checkpoint-14750.jsonl",
    "ft12_gpt-neo": "eval-ft-stage2-gpt-neo-GithubCodeDocStep1n2Filtered-Jan21-05-33-2022-checkpoint-1750.jsonl",
    "ft12_gpt-neo-27": "eval-ft-stage2-gpt-neo-27-GithubCodeDocStep1n2Filtered-Jan23-04-41-2022-checkpoint-4500.jsonl",
    "ft1_codeT5": "eval-ft-stage1-codeT5-GithubCodeDoc-Mar07-19-47-2022-checkpoint-420000.jsonl",
    "ft2_codeT5": "eval-ft-stage1-codeT5-GithubCodeDocStep1n2Filtered-Mar07-17-37-2022-checkpoint-41000.jsonl",
    "ft12_codeT5": "eval-ft-stage2-codeT5-GithubCodeDocStep1n2Filtered-Mar12-20-28-2022-checkpoint-14000.jsonl",
}
data_files = {k: os.path.join(data_dir, v) for k, v in data_files.items()}
for model in models:
    if model not in data_files:
        raise ValueError(f"Model {model} is not in the data files.")

# %% [markdown]
# ## Load the data
ref_docs, _, source_codes = load_code_doc(source_file)
assert len(ref_docs) == len(source_codes) == 2677
gen_docs = {}  # type: Dict[str, List[str]]
for model in models:
    gen_docs[model] = load_key_jsonl(data_files[model], key="gen")
    assert len(gen_docs[model]) == len(source_codes)

# %% [markdown]
# ## 1. merge results into one file
all_results = []  # type: List[Dict[str, str]]
for i, (source_code, ref_doc) in enumerate(zip(source_codes, ref_docs)):
    d = {"testset_id": i, "code": source_code, "reference": ref_doc}
    for model in models:
        d[model] = gen_docs[model][i]
    all_results.append(d)

# write back to data_dir
if args.replace or not os.path.exists(os.path.join(data_dir, "all_eval.jsonl")):
    with open(os.path.join(data_dir, "all_eval.jsonl"), "w") as f:
        for d in all_results:
            f.write(json.dumps(d) + "\n")
if args.replace or not os.path.exists(os.path.join(data_dir, "all_eval.csv")):
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(data_dir, "all_eval.csv"), index=True)
# %% [markdown]
# ## 2. select samples and collect codex results
# random select samples from all_results
# if not os.path.exists(os.join(save_dir, "raw_samples_human_eval.jsonl")):
select_ids = random.sample(range(len(all_results)), num_samples)
selected_results = [all_results[i] for i in select_ids]


# %% codex call
for i, d in enumerate(selected_results):
    print(f"call codex fro example No.{i}/{len(selected_results)}")
    codex_Py2NL = call_codex(d["code"], mode="Py2NL")
    codex_Py2Doc = call_codex(d["code"], mode="Py2Doc")
    print(
        f"codex_Py2NL: {codex_Py2NL} \n\n"
        f"codex_Py2Doc: {codex_Py2Doc} \n\ncode: {d['code']}"
    )
    d["codex_Py2NL"] = codex_Py2NL
    d["codex_Py2Doc"] = codex_Py2Doc
    # sleep 6 seconds due to the api limit
    time.sleep(9)

# %% [markdown]
# ## 3. save the selected results
if args.replace or not os.path.exists(
    os.path.join(save_dir, "raw_samples_human_eval.jsonl")
):
    with open(os.path.join(save_dir, "raw_samples_human_eval.jsonl"), "w") as f:
        for d in selected_results:
            f.write(json.dumps(d) + "\n")
if args.replace or not os.path.exists(
    os.path.join(save_dir, "raw_samples_human_eval.csv")
):
    df = pd.DataFrame(selected_results)
    df.to_csv(os.path.join(save_dir, "raw_samples_human_eval.csv"), index=True)

# %% [markdown]
# ## 4. prepare files for annotators
internal_dir = os.path.join(save_dir, "internal")
external_dir = os.path.join(save_dir, "external")
os.makedirs(internal_dir, exist_ok=True)
os.makedirs(external_dir, exist_ok=True)


def example2csv(
    data: Dict[str, str],
    file: TextIO,
    mode: Literal["interal", "external"],
) -> None:
    """
    save the example to csv file
    """
    if mode == "internal":
        header_cols = ["testset_id", "code"]
        write_model_name = True
        write_label_cols = False
    elif mode == "external":
        header_cols = ["code"]
        write_model_name = False
        write_label_cols = True
    else:
        raise ValueError("mode has to be 'internal' or 'external'")
    models = [
        "reference",
        "ft12_codeT5",
        "ft12_gpt-neo-27",
        "ft12_gpt-neo",
        "ft12_gpt2",
        "ft1_codeT5",
        "ft2_codeT5",
        "codex_Py2Doc",
        "codex_Py2NL",
    ]
    label_cols = [
        "step 1. explaining",
        "step 2. informative/coverage",
        "step 3. coherence/correctness",
        "step 4. readability/fluency",
        "step 5. format/style",
    ]

    output_data = []
    for model in models:
        row = {}
        for hc in header_cols:
            row[hc] = data[hc]
        if write_model_name:
            row["model"] = model
        row["docstring"] = data[model]
        if write_label_cols:
            for lc in label_cols:
                row[lc] = ""
        output_data.append(row)

    # write to csv
    df = pd.DataFrame(output_data)
    df.to_csv(file, index=False)


# %%
for i, d in enumerate(selected_results):
    print(f"prepare files for example No.{i}/{len(selected_results)}")
    file_name = f"{i}.csv"
    # internal
    internal_file_path = os.path.join(internal_dir, file_name)
    with open(internal_file_path, "w") as f:
        example2csv(d, f, mode="internal")
    # external
    external_file_path = os.path.join(external_dir, file_name)
    with open(external_file_path, "w") as f:
        example2csv(d, f, mode="external")

# %%
