# %%
import json
import os

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification,
    TextClassificationPipeline,
)

from typing import Iterable, List, Dict, Tuple

# %% settings
data_dir = "/mnt/default/data/"
step1_file = data_dir + "GithubCodeDocStep1Filtered.train.jsonl"
assert os.path.exists(step1_file), "step1_file not found"
step1_model_path = data_dir + "checkpoint-700"
step2_model_path = data_dir + "checkpoint-2400"
assert os.path.exists(step1_model_path), "step1_model_path not found"
assert os.path.exists(step2_model_path), "step2_model_path not found"
output_file = "/mnt/default/data/step2_filter_result.jsonl"

# %% load code doc
def load_code_doc(file, format="GithubCodeDocStep1Filtered"):
    if format == "GithubCodeDocStep1Filtered":
        with open(file, "r") as f:
            data = [json.loads(line) for line in f]
        codes = [d["code"] for d in data]
        docstrings = [d["docstring"] for d in data]
        complexities = [d["complexity"] for d in data]
    else:
        raise ValueError(f"format {format} not supported")
    return codes, docstrings, complexities


class Annotator:
    def __init__(self):
        self.tokenizer_step1 = BertTokenizerFast.from_pretrained(step1_model_path)
        self.model_step1 = BertForSequenceClassification.from_pretrained(
            step1_model_path, num_labels=1
        )
        self.tokenizer_step2 = BertTokenizerFast.from_pretrained(step2_model_path)
        self.model_step2 = BertForSequenceClassification.from_pretrained(
            step2_model_path, num_labels=5
        )
        self.model_step1.eval()
        self.model_step2.eval()
        # check available num of gpus
        if torch.cuda.device_count() > 1:
            # put model on device 0 and 1
            self.model_step1.to("cuda:0")
            self.model_step2.to("cuda:1")

        assert torch.cuda.device_count() > 1

    def logicscore(
        self,
        hypotheses: List[Iterable],
        batch_size=64,
        **kwargs,
    ):
        step1_scores = []
        # # put mode on device 0 if available
        # model_device = self.model_step1.device
        # if torch.cuda.is_available() and self.model_step1.device == "cpu":
        #     self.model_step1.to("cuda:0")

        for i in range(0, len(hypotheses), batch_size):
            print(f"{i}/{len(hypotheses)} logicscores generated")
            batch = hypotheses[i : i + batch_size]
            inputs = self.tokenizer_step1(
                batch, truncation=True, padding=True, return_tensors="pt"
            ).to(self.model_step1.device)
            with torch.no_grad():
                scores = self.model_step1(**inputs).logits.cpu().numpy()
                scores = scores * 3  # scale up to [0,3]
            scores = scores.flatten().tolist()
            step1_scores.extend(scores)

        # # put model back
        # self.model_step1.to(model_device)

        return step1_scores

    def step2_score(self, hypotheses: List[Iterable], batch_size=64, **kwargs):
        step2_cates = []

        for i in range(0, len(hypotheses), batch_size):
            print(f"{i}/{len(hypotheses)} step2 scores generated")
            batch = hypotheses[i : i + batch_size]
            inputs = self.tokenizer_step2(
                batch, truncation=True, padding=True, return_tensors="pt"
            ).to(self.model_step2.device)
            with torch.no_grad():
                preds = self.model_step2(**inputs).logits.cpu().numpy()
                cates = np.argmax(preds, axis=1)
            cates = cates.flatten().tolist()
            step2_cates.extend(cates)
        return step2_cates


# %% load
codes, docstrings, complexities = load_code_doc(step1_file)
doc_code_pairs = list(zip(docstrings, codes))

# %% setup and eval
annotator = Annotator()
step1_scores = annotator.logicscore(doc_code_pairs, batch_size=512)
step2_cates = annotator.step2_score(doc_code_pairs, batch_size=512)
# print(f"step1_scores: {step1_scores}")
# print(f"step2_cates: {step2_cates}")

# %% save jsonl
with open(output_file, "w") as f:
    for doc, code, step1_score, step2_cate, complexities in zip(
        docstrings, codes, step1_scores, step2_cates, complexities
    ):
        d = {
            "docstring": doc,
            "code": code,
            "step1_score": step1_score,
            "step2_cate": step2_cate,
            "complexity": complexities,
        }
        f.write(json.dumps(d) + "\n")

# %%
