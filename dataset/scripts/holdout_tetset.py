# %%
import json
import os
import time
from numbers import Number
from timeit import repeat
from typing import List, Tuple
from matplotlib import lines
import numpy as np
from torch import threshold
import transformers
from transformers import AutoModelForCausalLM
from datasets import load_dataset

from utils import load_code_doc

# # %%
# model = AutoModelForCausalLM.from_pretrained(
#     "EleutherAI/gpt-j-6B", cache_dir="/home/v-haotiancui/gpt-j"
# )

# %% [markdown]
# # Make holdout set

# %%
data_file_all = (
    "/home/v-haotiancui/NL2Code/Copilot-2/code2text"
    "/GPT-J/data/GithubCodeDoc.train.jsonl"
)
data_file_filtered = (
    "/home/v-haotiancui/NL2Code/Copilot-2/code2text"
    "/GPT-J/data/GithubCodeDocStep1n2Filtered.train.jsonl"
)
valid_split_percentage_all = 1
valid_split_percentage_filtered = 5
data_file_all_nopromt = (
    "/home/v-haotiancui/NL2Code/Copilot-2/code2text/"
    "codeT5/data/GithubCodeDoc.train.jsonl"
)
data_file_filtered_noprompt = (
    "/home/v-haotiancui/NL2Code/Copilot-2/code2text/"
    "codeT5/data/GithubCodeDocStep1n2Filtered.train.jsonl"
)
output_file = (
    "/home/v-haotiancui/NL2Code/Copilot-2/dataset/"
    "GithubCodeDocStep1n2Filtered.holdout.jsonl"
)
human_eval_file = (
    "/home/v-haotiancui/NL2Code/Copilot-2/dataset/"
    "ExplanationAnnotatedHighQuality.json"
)
test_file = (
    "/home/v-haotiancui/NL2Code/Copilot-2/dataset/"
    "GithubCodeDocStep1n2Filtered.test.jsonl"
)

# %%
def load_data(data_file, validation_split_percentage):
    data_files = {}
    data_files["train"] = data_file
    extension = "json"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{validation_split_percentage}%:]",
        )
    return raw_datasets


# %%
all_data = load_data(data_file_all, valid_split_percentage_all)
filtered_data = load_data(data_file_filtered, valid_split_percentage_filtered)

all_data_valid = all_data["validation"]
filtered_data_valid = filtered_data["validation"]


# %%
if not os.path.exists(output_file):
    filtered_valid_text = filtered_data_valid["text"]
    all_valid_text = all_data_valid["text"]
    indices_filtered2all = []
    for i, text in enumerate(filtered_valid_text):
        for j, text_all in enumerate(all_valid_text):
            if text == text_all:
                print(f"found {i} {j}")
                indices_filtered2all.append((i, j))
                break

    # find the raw data for holdout set
    source_data = data_file_filtered_noprompt
    raw_data = load_data(source_data, valid_split_percentage_filtered)
    raw_data_valid = raw_data["validation"]
    raw_valid_code = raw_data_valid["code"]
    for code, prompted_text in zip(raw_valid_code, filtered_valid_text):
        assert code == prompted_text[12 : len(code) + 12], (code, prompted_text)
    holdout_indices = [i for i, _ in indices_filtered2all]

    # write the holdout set to file
    holdout_dataset = raw_data_valid.select(holdout_indices)
    holdout_dataset.to_json(
        output_file,
        lines=True,
    )
    # test
    for i in range(len(holdout_dataset)):
        assert (
            holdout_dataset["code"][i]
            == raw_data_valid["code"][indices_filtered2all[i][0]]
        )

# %% [markdown]
# # Find the totally unleaked data
# 1. in the holdout set
# 2. in the human-eval set

# %%
def edit_distance(s1: str, s2: str) -> Number:
    from polyleven import levenshtein

    d = levenshtein(s1, s2)
    return d


# %%
all_data_nopromt = load_data(data_file_all_nopromt, valid_split_percentage_all)
filtered_data_noprompt = load_data(
    data_file_filtered_noprompt,
    valid_split_percentage_filtered,
)
all_data_nopromt_train = all_data_nopromt["train"]
filtered_data_noprompt_train = filtered_data_noprompt["train"]

all_train_codes = all_data_nopromt_train["code"]
all_train_docs = all_data_nopromt_train["docstring"]
filtered_train_codes = filtered_data_noprompt_train["code"]

# %% [markdown]
# # unleaked data on holdout set
# %%
def compute_distances(
    source_texts: List[str], target_texts: List[str]
) -> Tuple[np.ndarray, List[int]]:
    tik = time.time()
    threshold_length = 300
    # init distance matrix to -1
    # distances = -np.ones((len(source_texts), len(target_texts)))
    repeated_indices = []
    for i, source in enumerate(source_texts):
        len_source = len(source)
        min_d = np.inf
        tok = time.time()
        print(f"{i}/{len(source_texts)}, time: {tok - tik:.4f}s")
        tik = tok
        for j, target in enumerate(target_texts):
            len_target = len(target)
            if (len_target > 0.95 * len_source) and (len_target < 1.05 * len_source):
                d = edit_distance(
                    source[:threshold_length],
                    target[:threshold_length],
                )
                # distances[i, j] = d
                if d < min_d:
                    min_d = d
                # break on the first hit
                if d < min(threshold_length, len_source) * 0.05:
                    print(f"repeated {i} to {j}, distance: {d}/{len(source)} \n")
                    print(f"source: {source} \n\n\n target: {target} \n\n\n")
                    break
        if min_d < min(threshold_length, len_source) * 0.05:
            repeated_indices.append(i)
    return None, repeated_indices


def strip_code(code: str) -> str:
    return "\n".join([line.strip() for line in code.split("\n")])


# %%
if os.path.exists("./repeated_indices.json"):
    with open("./repeated_indices.json", "r") as f:
        repeated_indices = json.load(f)
else:
    repeated_indices = {}
holdout_docs, _, holdout_codes = load_code_doc(data_file=output_file)

# (
#     holdout_to_filtered_distances,
#     repeated_indices["holdout_to_filtered"],
# ) = compute_distances(holdout_codes, filtered_train_codes)
# (
#     holdout_to_all_distances,
#     repeated_indices["holdout_to_all"],
# ) = compute_distances(holdout_codes, all_train_codes)
holdout_docs_striped = [strip_code(code) for code in holdout_docs]
all_train_docs_striped = [strip_code(code) for code in all_train_docs]
(
    holdout_to_all_distances,
    repeated_indices_docs,
) = compute_distances(holdout_docs_striped, all_train_docs_striped)

repeated_indices["holdout_to_all"] = list(
    set(repeated_indices_codes + repeated_indices_docs)
)
# %% [markdown]
# # unleaked data on human-eval set
# %%
humaneval_docs, _, humaneval_codes = load_code_doc(data_file=human_eval_file)

# (
#     humaneval_to_filtered_distances,
#     repeated_indices["humaneval_to_filtered"],
# ) = compute_distances(humaneval_codes, filtered_train_codes)
humaneval_codes_striped = [strip_code(code) for code in humaneval_codes]
all_train_codes_striped = [strip_code(code) for code in all_train_codes]
(
    humaneval_to_all_distances,
    repeated_indices_codes,
) = compute_distances(humaneval_codes_striped, all_train_codes_striped)

humaneval_docs_striped = [strip_code(code) for code in humaneval_docs]
(
    humaneval_to_all_distances,
    repeated_indices_docs,
) = compute_distances(humaneval_docs_striped, all_train_docs_striped)

repeated_indices["humaneval_to_all"] = list(
    set(repeated_indices_codes + repeated_indices_docs)
)
# %% output repeated indices into json file
if not os.path.exists("./repeated_indices.json"):
    with open("./repeated_indices.json", "w") as f:
        json.dump(repeated_indices, f)

# %% [markdown]
# # Make test set

# %%

# read repeated indices
with open("./repeated_indices.json", "r") as f:
    repeated_indices = json.load(f)

# read json file
with open(human_eval_file, "r") as f:
    human_eval_data = json.load(f)
test_examples = []
for i, d in enumerate(human_eval_data):
    if i not in repeated_indices["humaneval_to_all"]:
        test_examples.append(d)
assert len(test_examples) == len(human_eval_data) - len(
    repeated_indices["humaneval_to_all"]
)

# read holdout set, jsonl file
with open(output_file, "r") as f:
    holdout_data = [json.loads(line) for line in f]
for i, d in enumerate(holdout_data):
    if i not in repeated_indices["holdout_to_all"]:
        test_examples.append(d)
assert len(test_examples) == (
    len(human_eval_data) - len(repeated_indices["humaneval_to_all"])
) + (len(holdout_data) - len(repeated_indices["holdout_to_all"]))
# %%
# output jsonl file
if not os.path.exists(test_file):
    with open(test_file, "w") as f:
        for d in test_examples:
            f.write(json.dumps(d) + "\n")

# %%
