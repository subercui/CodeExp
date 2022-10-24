# %% [markdown]
# # Finetune GPT-J using distributed training and save the model
# %%
import json
import os
import torch
import pandas as pd
from transformers import (
    DistilBertForSequenceClassification,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

# %% [markdown]
# ## Load the model
# %% test with GPT-NEO, comment out if you want to use GPT-J
from transformers import DataCollatorWithPadding, GPT2Tokenizer, GPTNeoForCausalLM

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
model.half()
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# %% [markdown]
# ## Inference interface
# %%
def generate(prompts: list, length=128):
    input = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    gen_tokens = model.generate(
        **input,
        do_sample=True,
        temperature=0.9,
        max_length=length,
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text


# print("Example of model generation:", generate(["I am a student."]))
# %% [markdown]
# ## Load the dataset
# %%
data_file = (
    "/home/v-haotiancui/NL2Code/Copilot-2/dataset/QualifiedExplanations.train.csv"
)
df = pd.read_csv(data_file)
source_texts, target_texts = (
    df["reference_code"].tolist(),
    df["description"].tolist(),
)

from sklearn.model_selection import train_test_split

train_source, val_source, train_target, val_target = train_test_split(
    source_texts, target_texts, test_size=0.1, random_state=42
)

assert len(train_source) > 200
print(f"num of training samples: {len(train_source)}")
print(f"num of validation samples: {len(val_source)}")
# %% tokenize
train_input_encodings = tokenizer(train_source, padding=False, truncation=True)
val_input_encodings = tokenizer(val_source, padding=False, truncation=True)
with tokenizer.as_target_tokenizer():
    train_labels = tokenizer(train_target, padding=False, truncation=True)["input_ids"]
    val_labels = tokenizer(val_target, padding=False, truncation=True)["input_ids"]
# %%
class ExplanationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ExplanationDataset(train_input_encodings, train_labels)
val_dataset = ExplanationDataset(val_input_encodings, val_labels)
# %% fine-tune
import numpy as np
from datasets import load_metric

# Metric
import sacrebleu

metric = load_metric("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=tokenizer.pad_token_id,
    pad_to_multiple_of=None,
)


# %%
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # output directory
    evaluation_strategy="steps",
    eval_steps=10,
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,  # batch size for evaluation
    # warmup_ratio=0.1,                 # ratio of learning rate for warmup
    # do not use warmup_steps if warmup_ratio is set
    # warmup_steps=500,                # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,  # initial learning rate
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
    load_best_model_at_end=True,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    tokenizer=tokenizer,  # tokenizer
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

trainer.train()
# %% [markdown]
# # GPT-J setup
# # %%
# from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt-j-6B")

# # %% [markdown]
# # # Distributed training setup
# # %%
# # Log on each process the small summary:
# logger.warning(
#     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
# )
# logger.info(f"Training/evaluation parameters {training_args}")
