# %% [markdown]
# # this script provides machine learning metrics and corresponding models using the annotated data.
# This script follows the colab notebook [Fine-tune ALBERT for sentence-pair classification](https://colab.research.google.com/github/NadirEM/nlp-notebooks/blob/master/Fine_tune_ALBERT_sentence_pair_classification.ipynb)
# from huggingface v3.2.0 [notebooks](https://huggingface.co/transformers/v3.2.0/notebooks.html)
# and [Processing sentence pairs](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/sentence_pairs_pt.ipynb#scrollTo=obgDzQhRpNzA).

# [multi-label classification](https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb)

# %% [markdown]
# we have multiple options to create the model to mimic human-eval logit score:
# 1. only input the docstring
# 2. input pairs of docstring and code, use as two sentences for bert like models
# 3. input pairs of docstring and code, use as one sentence
# here we simply use the first option.
# TODO: try later options
# %%
import os
import datetime
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification,
)
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# %% configs
file_path = "/home/v-haotiancui/NL2Code/Copilot-2/dataset/ExplanationAnnotated.json"
model_path = "bert-base-uncased"
time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# %% load data
def load_data(path, raw: bool = False):
    """
    Loads the data from the given path.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if raw:
        return data
    docstrings = [d["docstring"] for d in data]
    codes = [d["code"] for d in data]
    logic_scores = [float(d["step1"]) / 3 for d in data]  # convert to [0, 1]
    step2_scores = [
        int(d["step2"]) if len(d["step2"]) > 0 else 4 for d in data
    ]  # 5 categories
    return docstrings, codes, logic_scores, step2_scores


docstrings, codes, _, step2_scores = load_data(file_path)
code_doc_pairs = list(zip(docstrings, codes))


# %% [markdown]
# # train BERT classifier with the trainer api
# %%
train_texts, train_labels = code_doc_pairs, step2_scores
print(f"num of all samples: {len(train_texts)}")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.15, random_state=42
)
print(f"num of training samples: {len(train_texts)}")
print(f"num of validation samples: {len(val_texts)}")
test_texts, test_labels = val_texts, val_labels

# %% [markdown]
# ## preprocessing
# %%
tokenizer = BertTokenizerFast.from_pretrained(model_path)
# Padding is set to false. We add tokenizer option to the trainer later and the
# input will be padded to the max length there.
# TODO: maybe the trainer could handle tokeniztion within itself, and we should
# only input the raw text here. Try this approach.
train_encodings = tokenizer(train_texts, truncation=True, padding=False)
val_encodings = tokenizer(val_texts, truncation=True, padding=False)
test_encodings = tokenizer(test_texts, truncation=True, padding=False)


class DocstringDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings["input_ids"]) == len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.int)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = DocstringDataset(train_encodings, train_labels)
val_dataset = DocstringDataset(val_encodings, val_labels)
test_dataset = DocstringDataset(test_encodings, test_labels)

# %% [markdown]
# ## Fine-tuning with Trainer
# %%
# Define metric classification accuracy
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": np.mean(preds == p.label_ids)}


training_args = TrainingArguments(
    report_to="wandb",
    run_name="ml-annotator-step2" + time_stamp,
    output_dir="./results_step2" + time_stamp,  # output directory
    logging_strategy="steps",
    logging_dir="./logs",  # directory for storing logs
    logging_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    # warmup_ratio=0.1,                 # ratio of learning rate for warmup
    # do not use warmup_steps if warmup_ratio is set
    # warmup_steps=500,                # number of warmup steps for learning rate scheduler
    learning_rate=1e-5,  # initial learning rate
    weight_decay=0.01,  # strength of weight decay
    load_best_model_at_end=True,
    metric_for_best_model="acc",
    greater_is_better=True,
)

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    tokenizer=tokenizer,  # tokenizer
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

# %% actual run
trainer.train()

# try predict on test set
predictions, label_ids, metrics = trainer.predict(test_dataset)
# %% [markdown]
# # apply model on the examples
inputs = tokenizer(docstrings[:6], truncation=True, padding=True, return_tensors="pt")
model(**inputs.to(model.device)).logits.detach().cpu().numpy()

from transformers import TextClassificationPipeline

classifier = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=model.device.index if model.device.type == "cuda" else -1,
    function_to_apply="none",
)
# FIXME: currently none but not sigmoid returns the correct prediction. This
# indicates that the model in training stage does not have the sigmoid activation.
# Need to fix the output layer during training.
classifier(docstrings[0])
classifier(docstrings[:6])

# %%
