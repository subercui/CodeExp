# %% [markdown]
# # Generate (filtered) datasets for finetuning
# The script by default runst the step 1 filtering and step 2 filtering on all
# collected data. If this is not what you want, you can:
# 1. skip all the steps after the section **Generate the step 1 auto filtered data**
# 2. run make_filtered_data() to generate the filtered data
# 3. run make_all_data() to generate the unfiltered data
# %%
from operator import index
from rouge import Rouge
import json
import os
import pandas as pd
from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor
from utils import merge_jsonl_files

from typing import List, Dict, Tuple


# %%
configs = {
    "dataset": "filtered",  # "all" or "filtered"
    "overwrite": False,  # whether to overwrite the output file
    "format": "T5",  # "GPT" or "T5", folowing code by default assumes "GPT"
}
DATASET = configs["dataset"]
current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)
output_dir = (
    "/home/v-haotiancui/NL2Code/Copilot-2/code2text/GPT-J/data/"
    if configs["format"] == "GPT"
    else "/home/v-haotiancui/NL2Code/Copilot-2/code2text/codeT5/data/"
)
if DATASET == "all":  # this is all collected data
    raw_file = (
        "/home/v-haotiancui/NL2Code/Copilot-2/dataset/all_processed_code_doc.jsonl"
    )
    output_file = output_dir + "GithubCodeDoc.train.jsonl"
elif DATASET == "filtered":  # this is the 30k data after filtering
    raw_file = "/home/v-haotiancui/NL2Code/Copilot-2/dataset/GithubCodeDocStep1FilteredStep2Labeled.jsonl"
    output_file = output_dir + "GithubCodeDocStep1n2Filtered.train.jsonl"
elif DATASET == "stars_gt60":
    raw_file = "/home/v-haotiancui/NL2Code/Copilot-2/dataset/pycodesuggest/scripts_by_code_doc_corpus/data_stars_gt60/code_doc.jsonl"
else:
    raise ValueError("DATASET should be 'all' or 'stars_gt60'")


# %%
def check_data():
    if DATASET == "all":
        jsonl_files = [
            "/home/v-haotiancui/NL2Code/Copilot-2/dataset/pycodesuggest/scripts_by_code_doc_corpus/data_folks_gt184/code_doc.jsonl",
            "/home/v-haotiancui/NL2Code/Copilot-2/dataset/pycodesuggest/scripts_by_code_doc_corpus/data_stars_gt60/code_doc.jsonl",
            "/home/v-haotiancui/NL2Code/Copilot-2/dataset/pycodesuggest/scripts_by_code_doc_corpus/data_uncloned/code_doc.jsonl",
            "/home/v-haotiancui/NL2Code/Copilot-2/dataset/pycodesuggest/scripts_by_code_doc_corpus/data_stars_gt103/code_doc.jsonl",
            "/home/v-haotiancui/NL2Code/Copilot-2/dataset/pycodesuggest/scripts_by_code_doc_corpus/data_stars_gt148/code_doc.jsonl",
            "/home/v-haotiancui/NL2Code/Copilot-2/dataset/pycodesuggest/scripts_by_code_doc_corpus/data_stars_gt270/code_doc.jsonl",
        ]
        merge_jsonl_files(jsonl_files, raw_file, overwrite=configs["overwrite"])
    else:
        raise FileNotFoundError("data file not found")


# %% [markdown]
# # Read raw files
# %%
def load_code_doc(raw_file: str) -> Tuple[List[str], List[str]]:
    # read jsonl file
    with open(raw_file, "r") as f:
        data = [json.loads(line) for line in f]

    if DATASET == "all":
        # Generate the training data format
        codes = [d["funcdef"] + "\n" + d["funcbody"] for d in data]
        docstrings = [d["docstring"] for d in data]
        return codes, docstrings
    elif DATASET == "filtered":
        with open(raw_file, "r") as f:
            data = [json.loads(line) for line in f]
        codes = [d["code"] for d in data]
        docstrings = [d["docstring"] for d in data]
        step1_scores = [d["step1_score"] for d in data]
        step2_cates = [d["step2_cate"] for d in data]
        complexities = [d["complexity"] for d in data]
        return codes, docstrings, step1_scores, step2_cates, complexities


# %% make_promts
def make_promts(source_texts, target_texts):
    texts = []
    for i in range(len(source_texts)):
        texts.append(
            "# Python 3 \n"
            + source_texts[i]
            + '\n\n"""Explanation of what the code does: \n'
            + target_texts[i]
            + '\n"""'
        )
    return texts


# %% make_filtered_data
def make_filtered_data(configs):
    assert DATASET == "filtered"
    codes, docstrings, step1_scores, step2_cates, complexities = load_code_doc(raw_file)
    step1_thres = 1.0
    step2_thres = 1
    filtered_data = []
    texts = make_promts(codes, docstrings)
    for i in range(len(codes)):
        if step1_scores[i] > step1_thres and step2_cates[i] > step2_thres:
            try:
                if configs["format"] == "GPT":
                    texts[i].encode("utf-8")
                    filtered_data.append(
                        {
                            "text": texts[i],
                            "index": i,
                            "step1_score": step1_scores[i],
                            "step2_cate": step2_cates[i],
                            "complexity": complexities[i],
                        }
                    )
                elif configs["format"] == "T5":
                    codes[i].encode("utf-8")
                    docstrings[i].encode("utf-8")
                    filtered_data.append(
                        {
                            "code": codes[i],
                            "docstring": docstrings[i],
                            "index": i,
                            "step1_score": step1_scores[i],
                            "step2_cate": step2_cates[i],
                            "complexity": complexities[i],
                        }
                    )
                else:
                    raise ValueError("format should be 'GPT' or 'T5'")
            except UnicodeEncodeError:
                print(f"UnicodeEncodeError when parsing: {texts[i]}")

    # save jsonl file
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            for d in filtered_data:
                f.write(json.dumps(d) + "\n")


def make_all_data(configs):
    assert DATASET == "all"
    codes, docstrings = load_code_doc(raw_file)

    data = []
    if configs["format"] == "GPT":
        texts = make_promts(source_texts=codes, target_texts=docstrings)
        for text in texts:
            try:
                text.encode("utf-8")  # make sure it can be encoded with utf-8
                data.append({"text": text})
            except:
                print("Error encoding utf-8: ", text)
    elif configs["format"] == "T5":
        for i in range(len(codes)):
            try:
                codes[i].encode("utf-8")
                docstrings[i].encode("utf-8")
                data.append(
                    {
                        "code": codes[i],
                        "docstring": docstrings[i],
                        "index": i,
                    }
                )
            except:
                print("Error encoding utf-8: ", codes[i])
    else:
        raise ValueError("format should be 'GPT' or 'T5'")

    # save jsonl file
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    # test loading data
    with open(output_file, "r") as f:
        data = [json.loads(line) for line in f]


# %% [markdown]
# # Generate the step 1 auto filtered data
# %%
assert DATASET == "all"
codes, descs = load_code_doc(raw_file)

# %% [markdown]
# # filter length and complexity
# %%
# filter out docstrings with params
def remove_param(desc):
    lines = desc.split("\n")
    new_lines = [line for line in lines if ":param" not in line]
    new_lines = [line for line in new_lines if ":arg" not in line]
    new_lines = [line for line in new_lines if not line.startswith(":")]
    new_lines = [line for line in new_lines if not line.startswith("@")]
    return new_lines


# %%
long_code = []
long_desc = []
complexities = []
for i, code in enumerate(codes):
    try:
        d = descs[i]
        d = d[: d.find(">>>")] if d.find(">>>") >= 0 else d
        if (
            len(remove_param(d)) >= 3
            and len(code.split("\n")) >= 6
            and len(code.split("\n")) <= 30
        ):
            # complexity messure
            v = cc_visit(code)
            if v[0].complexity > 3:
                long_code.append(code)
                long_desc.append(d)
                complexities.append(v[0].complexity)
    except:
        print("Error occured: ", i)
        continue

# %% [markdown]
# # Store data in jsonl file
# %%
file_name = (
    "/home/v-haotiancui/NL2Code/Copilot-2/dataset/"
    "GithubCodeDocStep1Filtered.train.jsonl"
)
assert len(long_code) == len(long_desc) == len(complexities)
with open(file_name, "w") as f:
    for i, code in enumerate(long_code):
        try:
            code.encode("utf-8")  # make sure it can be encoded with utf-8
            f.write(
                json.dumps(
                    {
                        "code": code,
                        "docstring": long_desc[i],
                        "complexity": complexities[i],
                    }
                )
                + "\n"
            )
        except:
            print("Error encoding utf-8: ", code)

# %% [markdown]
# # Step 2. filter using the ml annotators

# %%
step1_file = file_name
assert os.path.exists(step1_file)
# load data
with open(step1_file, "r") as f:
    data = [json.loads(line) for line in f]
codes = [d["code"] for d in data]
docstrings = [d["docstring"] for d in data]

# %%
