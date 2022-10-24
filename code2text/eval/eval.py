# %% imports and settings
import os
import sys
import json
import itertools
import operator
from collections import Counter
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
from numbers import Number
import argparse
from IPython import get_ipython

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    BertTokenizerFast,
    BertForSequenceClassification,
)
import nltk
from nltk.translate.bleu_score import *
from nltk.translate import meteor
from nltk import word_tokenize
import re
from rouge import Rouge
from bert_score import BERTScorer

sys.path.append("/home/v-haotiancui/NL2Code/Copilot-2/dataset/scripts/")
from utils import (
    isnotebook,
    kendalls_tau,
    rouge_score,
    nltk_bleu,
    meteor_score,
    bertscore,
    codebertscore,
    fact_scores,
)


current_dir = os.path.dirname(os.path.realpath(__file__))

# instruct what to do in this script
parser = argparse.ArgumentParser()
parser.add_argument(
    "mode",
    choices=["generate", "eval", "both", "compare"],
    default="eval",
    help="generate: generate on test set using gpt-neo;"
    " eval: evaluate existing result files; "
    "compare: compute kendal's tau to human-eval.",
)
parser.add_argument(
    "-s",
    "--save",
    action="store_true",
    default=False,
    required=False,
    help="save the model. Only used in the generate mode",
)
parser.add_argument(
    "--level",
    choices=["corpus", "sentence"],
    default="corpus",
    required=False,
    help="level of the model to evaluate",
)
parser.add_argument(
    "--by-chunk", action="store_true", default=False, help="generate in by chunk mode."
)
parser.add_argument(
    "--fine-tuned",
    action="store_true",
    default=False,
    help="generate using the fine-tuned model",
)
if isnotebook():
    # args = parser.parse_args(args=["compare", "--level", "sentence"])
    args = parser.parse_args(args=["generate", "--fine-tuned"])
elif __name__ != "__main__":
    args = parser.parse_args(args=["compare", "--level", "sentence"])
else:
    args = parser.parse_args()

BY_CHUNKS = args.by_chunk
FINE_TUNED = args.fine_tuned
CODEX = False
print(f"By_chunk: {BY_CHUNKS}")
print(f"Fine_tuned: {FINE_TUNED}")
print(f"save: {args.save}")


# %% sucessfully load the model
def load_gptneo(FINE_TUNED=False):
    if FINE_TUNED:
        # model_path = "/home/v-haotiancui/blobfuse/euphillyblob-amulet/checkpoints0804-fp32/checkpoint-3000"
        # model_path = "/home/v-haotiancui/NL2Code/Copilot-2/code2text/GPT-J/results/gpt-neo/checkpoint-3000"
        # model_path = (
        #     "/home/v-haotiancui/NL2Code/Copilot-2/code2text/"
        #     "GPT-J/results/Explanation-gpt-neo/checkpoint-2500"
        # )

        # model_path = (
        #     "/home/v-haotiancui/NL2Code/Copilot-2/code2text/"
        #     "GPT-J/results/ExplanationPlusConala-gpt-neo/checkpoint-2500"
        # )

        # model_path = (
        #     "/home/v-haotiancui/NL2Code/Copilot-2/code2text/"
        #     "GPT-J/results/ft-stage1-gpt-neo-GithubCodeDoc"
        # )

        # model_path = (
        #     "/home/v-haotiancui/NL2Code/Copilot-2/code2text/"
        #     "GPT-J/results/finetune2-ft-stage2-gpt-neo-Explanations"
        # )

        model_path = (
            "/home/v-haotiancui/NL2Code/Copilot-2/code2text/"
            "GPT-J/results/finetune2-ft-stage2-gpt2-Explanations"
        )
        print(f"Loading fine-tuned model from {model_path}")
    else:
        model_path = "EleutherAI/gpt-neo-1.3B"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # model = model.half()
    model = model.to("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# %% load data
# human-eval results
human_eval = {
    "human_eval_codex": [
        2.75,
        2.75,
        2.5,
        3,
        2.75,
        0.5,
        2.5,
        3,
        2,
        0.5,
        1.75,
        0.75,
        0.5,
        3,
        0.75,
        0.5,
        3,
        1.25,
        3,
    ],
    "human_eval_codex(by_chunk)": [
        1.75,
        2,
        2,
        2.25,
        2.25,
        2.5,
        2.5,
        2.5,
        2,
        2.25,
        1.75,
        0.75,
        1,
        2.25,
        1.25,
        0.5,
        2.5,
        1.25,
        3,
    ],
    "human_eval_gpt-neo": [
        1.75,
        2.5,
        0.75,
        2.75,
        0,
        2.25,
        0,
        0.75,
        2.25,
        0.25,
        0.5,
        1.125,
        1.75,
        0.75,
        0.5,
        1.75,
        0.5,
        1.5,
        2,
    ],
    "human_eval_gpt-neo(by_chunk)": [
        1,
        1.5,
        1.5,
        1.5,
        1.75,
        1,
        1.75,
        1.75,
        1.5,
        0.75,
        0.5,
        0.75,
        1.75,
        1.75,
        0.25,
        0.75,
        1,
        1.75,
        2.25,
    ],
    "human_eval_gpt-neo(fine_tuned)": [
        1.5,
        2.125,
        1.75,
        2,
        1.75,
        2.75,
        2.5,
        2.5,
        3,
        2.5,
        2,
        2,
        2.25,
        2.5,
        1.75,
        1.5,
        2.25,
        1.75,
        2.5,
    ],
    "human_eval_gpt-neo(fine_tuned)(by_chunk)": [
        2,
        2.25,
        2.25,
        1.5,
        1.75,
        2,
        2.75,
        2.25,
        1.75,
        2.25,
        1.75,
        1.75,
        2,
        2.5,
        1.75,
        1.75,
        2.25,
        1.5,
        2.5,
    ],
}

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


# %% make prompt
def make_prompt(codes, block_codes, BY_CHUNKS):
    def decorate(code):
        prompt = "# Python 3 \n" + code + '\n\n"""Explanation of what the code does: \n'
        return prompt

    if not BY_CHUNKS:
        prompts = [decorate(code) for code in codes]
        return prompts, None
    else:
        prompts = []
        code_start_end = []
        count = 0
        for blocks in block_codes:
            start = count
            count += len(blocks)
            end = count
            prompts.extend([decorate(code) for code in blocks])
            code_start_end.append((start, end))
        assert len(prompts) == code_start_end[-1][1]
    return prompts, code_start_end


# %% [markdown]
# # Generate and save texts
# %%
def generate_gpt_neo_pipeline(save=False):
    # load model
    model, tokenizer = load_gptneo(FINE_TUNED)
    # load data
    docs, block_codes, codes = load_code_doc()
    # make prompt
    prompts, code_start_end = make_prompt(codes, block_codes, BY_CHUNKS)
    # generate texts
    def generate(prompts, length=256):
        # FIXME: remove the length restriction
        prompts = [p[-500:] for p in prompts]
        # input = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")

        length = len(input["input_ids"][0]) + 100
        print(length)
        gen_tokens = model.generate(
            **input,
            do_sample=True,
            temperature=0.9,
            max_length=length,
        )
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return gen_text

    batch_size = 4
    res = []
    if BY_CHUNKS:
        gen_texts = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            gen_text = generate(batch)
            print("\n\n".join(gen_text))
            for j, text in enumerate(gen_text):
                gen_text[j] = text[text.find("code does: \n") + 12 :]
            gen_texts.extend(gen_text)
        for i, start_end in enumerate(code_start_end):
            start, end = start_end
            text = "\n".join(gen_texts[start:end])
            d = {"gen": text}
            res.append(d)
    else:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            gen_text = generate(batch)
            print("\n\n".join(gen_text))
            for j, text in enumerate(gen_text):
                d = {
                    "gen": text[text.find("code does: \n") + 12 :],
                    "desc": batch[j],
                }
                res.append(d)
    if save:
        with open(
            os.path.join(
                current_dir,
                f"generated-ft-stage2-gpt2-Explanations.jsonl",
                # f"generated-finetune_{FINE_TUNED}-by_chunks_{BY_CHUNKS}.jsonl",
            ),
            "w",
        ) as f:
            for d in res:
                f.write(json.dumps(d) + "\n")
    return res


# %% [markdown]
# # Evaluation ML score
# %%
def logicscore(hypotheses, references, level="corpus", batch_size=24, **kwargs):
    model.eval()
    total_score = []

    for i in range(0, len(hypotheses), batch_size):
        batch = hypotheses[i : i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            scores = model(**inputs).logits.cpu().numpy()
        scores = scores.flatten().tolist()
        total_score.extend(scores)

    if level != "corpus":
        return total_score

    avg_score = sum(total_score) / len(total_score)
    print("logic avg_score: %.4f" % avg_score)
    return avg_score


# %% grammar check
if args.mode == "eval":
    import language_tool_python

    tool = language_tool_python.LanguageTool("en-US")

    def grammar_check(hypotheses, references, normalize=True, **kwargs):
        count = 0
        total_score = 0.0

        for i, hyp in enumerate(hypotheses):
            if len(hyp) == 0 or len(hyp.split()) == 0:
                continue
            matches = tool.check(hyp)
            if normalize:
                score = len(matches) / len(hyp.split())
            else:
                score = len(matches)
            total_score += score
            count += 1

        avg_score = total_score / count
        print("grammar_check avg_score: %.4f" % avg_score)
        return avg_score


# %% branching keywords for branch coverage eval


def branching_keywords(hypotheses, references, normalize=True, **kwargs):
    keywords = set(
        [
            "if",
            "else",
            "try",
            "except",
            "raise",
            "case",
            "default",
            "then",
            "otherwise",
            "True",
            "true",
            "False",
            "false",
            "when",
        ]
    )
    count = 0
    total_score = 0.0

    for i, hyp in enumerate(hypotheses):
        if len(hyp) == 0 or len(hyp.split()) == 0:
            continue
        if normalize:
            score = sum([len(re.findall(w, hyp)) for w in keywords]) / len(hyp.split())
        else:
            score = sum([len(re.findall(w, hyp)) for w in keywords])

        total_score += score
        count += 1

    avg_score = total_score / count
    print("branching_keywords avg_score: %.4f" % avg_score)
    return avg_score


# %% evaluate pipeline
def read_files(to_eval=None, **kwargs):
    if to_eval and isinstance(to_eval, str):
        assert to_eval.endswith(".jsonl"), "to_eval must be a jsonl file"
        assert os.path.exists(to_eval), f"{to_eval} does not exist"
        print(f"Reading file {to_eval}")
        with open(to_eval, "r") as f:
            gen_data = [json.loads(line) for line in f]
        gen_docs = [d["gen"] for d in gen_data]
        return gen_docs
    if not CODEX:
        with open(
            os.path.join(
                current_dir,
                f"generated-finetune_{FINE_TUNED}-by_chunks_{BY_CHUNKS}.jsonl",
            ),
            "r",
        ) as f:
            gen_data = [json.loads(line) for line in f]
        gen_docs = [d["gen"] for d in gen_data]
    else:
        # read codex csv
        if BY_CHUNKS:
            with open(
                "/home/v-haotiancui/NL2Code/Copilot-2/code2text/Codex/new-codex-genetated-rouge-1-p_0.2089-r0.2886_-f_0.2238-rouge-l-p_0.1934-r0.2681_-f_0.2076.chunks.csv",
                # "/home/v-haotiancui/NL2Code/Copilot-2/code2text/eval/generated-finetune_explanation.jsonl",
                "r",
                encoding="utf-8",
            ) as f:
                # gen_data = [json.loads(line) for line in f]
                codex = pd.read_csv(f)
        else:
            with open(
                "/home/v-haotiancui/NL2Code/Copilot-2/code2text/Codex/new-codex-genetated-rouge-1-p_0.2876-r0.2229_-f_0.2295-rouge-l-p_0.2680-r0.2066_-f_0.2131.csv",
                # "/home/v-haotiancui/NL2Code/Copilot-2/code2text/eval/generated-finetune_explanation-plus-conala.jsonl",
                "r",
                encoding="utf-8",
            ) as f:
                codex = pd.read_csv(f)
                # gen_data = [json.loads(line) for line in f]
        # gen_docs = [d["gen"] for d in gen_data]
        gen_docs = codex["generated_description"].tolist()
    return gen_docs


def eval_multi_metrics(
    gendocs: Iterable[str],
    docs: Iterable[str],
    codes: Iterable[str],
    metrics: Iterable[Callable[..., Optional[List]]],
    normalize: bool = False,
    level: Literal["corpus", "sentence"] = "corpus",
) -> Dict[str, List]:
    """Evaluates multiple metrics on a corpus of documents."""
    scores = {}
    for metric in metrics:
        res = metric(gendocs, docs, codes=codes, normalize=normalize, level=level)
        if res:
            scores[metric.__name__] = res
    return scores


def eval_pipeline(save=False, level="corpus"):
    data_file = "/home/v-haotiancui/NL2Code/Copilot-2/dataset/GithubCodeDocStep1n2Filtered.test.jsonl"
    # data_file = "/home/v-haotiancui/NL2Code/Copilot-2/dataset/ExplanationAnnotatedHighQuality.json"
    # data_file = "/home/v-haotiancui/NL2Code/Copilot-2/dataset/QualifiedExplanations.chunks.valid.jsonl"
    eval_dir = (
        "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_3k_testset/"
    )
    eval_items = {
        "eval_codeT5-base-multi-sum": (
            eval_dir + "eval-Salesforce-codet5-base-multi-sum.jsonl"
        ),
        "eval_gpt-neo": (eval_dir + "eval-EleutherAI-gpt-neo-1.3B.jsonl"),
        "eval_gpt-neo-27": (eval_dir + "eval-EleutherAI-gpt-neo-2.7B.jsonl"),
        "eval_ft1_gpt2": (
            eval_dir
            + "eval-ft-stage1-gpt2-GithubCodeDoc-Jan21-07-58-2022-checkpoint-157500.jsonl"
        ),
        "eval_ft2_gpt2": (
            eval_dir
            + "eval-ft-stage1-gpt2-GithubCodeDocStep1n2Filtered-Jan21-00-31-2022-checkpoint-14750.jsonl"
        ),
        "eval_ft12_gpt2": (
            eval_dir
            + "eval-ft-stage2-gpt2-GithubCodeDocStep1n2Filtered-Jan21-13-24-2022-checkpoint-14750.jsonl"
        ),
        "eval_ft1_gpt-neo": (
            eval_dir
            + "eval-ft-stage1-gpt-neo-GithubCodeDoc-Jan16-16-49-2022-checkpoint-80000.jsonl"
        ),
        "eval_ft2_gpt-neo": (
            eval_dir
            + "eval-ft-stage1-gpt-neo-GithubCodeDocStep1n2Filtered-Jan20-07-50-2022-checkpoint-4750.jsonl"
        ),
        "eval_ft12_gpt-neo": (
            eval_dir
            + "eval-ft-stage2-gpt-neo-GithubCodeDocStep1n2Filtered-Jan21-05-33-2022-checkpoint-1750.jsonl"
        ),
        "eval_ft1_gpt-neo-27": (
            eval_dir
            + "eval-ft-stage1-gpt-neo-27-GithubCodeDoc-Jan17-05-10-2022-checkpoint-55000.jsonl"
        ),
        "eval_ft2_gpt-neo-27": (
            eval_dir
            + "eval-ft-stage1-gpt-neo-27-GithubCodeDocStep1n2Filtered-Jan22-02-47-2022-checkpoint-9500.jsonl"
        ),
        "eval_ft12_gpt-neo-27": (
            eval_dir
            + "eval-ft-stage2-gpt-neo-27-GithubCodeDocStep1n2Filtered-Jan23-04-41-2022-checkpoint-4500.jsonl"
        ),
        "eval_ft1_codeT5": (
            eval_dir
            + "eval-ft-stage1-codeT5-GithubCodeDoc-Mar07-19-47-2022-checkpoint-420000.jsonl"
        ),
        "eval_ft2_codeT5": (
            eval_dir
            + "eval-ft-stage1-codeT5-GithubCodeDocStep1n2Filtered-Mar07-17-37-2022-checkpoint-41000.jsonl"
        ),
        "eval_ft12_codeT5": (
            eval_dir
            + "eval-ft-stage2-codeT5-GithubCodeDocStep1n2Filtered-Mar12-20-28-2022-checkpoint-14000.jsonl"
        ),
        # "eval_ft1_gpt2": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage1-gpt2-GithubCodeDoc-Jan21-07-58-2022.jsonl",
        # "eval_ft2_gpt2": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage1-gpt2-GithubCodeDocStep1n2Filtered-Jan21-00-31-2022.jsonl",
        # "eval_ft12_gpt2": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage2-gpt2-GithubCodeDocStep1n2Filtered-Jan21-13-24-2022.jsonl",
        # "eval_ft1_gpt-neo": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage1-gpt-neo-GithubCodeDoc-Jan16-16-49-2022.jsonl",
        # "eval_ft2_gpt-neo": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage1-gpt-neo-GithubCodeDocStep1n2Filtered-Jan20-07-50-2022.jsonl",
        # "eval_ft12_gpt-neo": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage2-gpt-neo-GithubCodeDocStep1n2Filtered-Jan21-05-33-2022.jsonl",
        # "eval_ft1_gpt-neo-27": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage1-gpt-neo-27-GithubCodeDoc-Jan17-05-10-2022.jsonl",
        # "eval_ft2_gpt-neo-27": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage1-gpt-neo-27-GithubCodeDocStep1n2Filtered-Jan22-02-47-2022.jsonl",
        # "eval_ft12_gpt-neo-27": "/home/v-haotiancui/blobfuse/euphillyblob-amulet/eval_results/on_140_data/eval-ft-stage2-gpt-neo-27-GithubCodeDocStep1n2Filtered-Jan23-04-41-2022.jsonl",
        #
        # "ft-stage1-gpt-neo-GithubCodeDoc": "/home/v-haotiancui/NL2Code/Copilot-2/code2text/eval/generated-ft-stage1-gpt-neo-GithubCodeDoc.jsonl",
        # "ft-stage2-gpt-neo-Explanations": "/home/v-haotiancui/NL2Code/Copilot-2/code2text/eval/generated-ft-stage2-gpt-neo-Explanations.jsonl",
        # "ft-stage2-gpt2-Explanations": "/home/v-haotiancui/NL2Code/Copilot-2/code2text/eval/generated-ft-stage2-gpt2-Explanations.jsonl",
        # "pretrained-gpt2-Explanations": "/home/v-haotiancui/NL2Code/Copilot-2/code2text/eval/generated-pretrained-gpt2-Explanations.jsonl",
        # "codex": {"CODEX": True, "BY_CHUNKS": False},
        # "codex(by_chunk)": {"CODEX": True, "BY_CHUNKS": True},
        # "gpt-neo": {"CODEX": False, "BY_CHUNKS": False, "FINE_TUNED": False},
        # "gpt-neo(by_chunk)": {"CODEX": False, "BY_CHUNKS": True, "FINE_TUNED": False},
        # "gpt-neo(fine_tuned)": {"CODEX": False, "BY_CHUNKS": False, "FINE_TUNED": True},
        # "gpt-neo(fine_tuned)(by_chunk)": {
        #     "CODEX": False,
        #     "BY_CHUNKS": True,
        #     "FINE_TUNED": True,
        # },
    }
    eval_metrics = [
        bertscore,
        codebertscore,
        rouge_score,
        nltk_bleu,
        meteor_score,
        fact_scores,
        # logicscore,
        # branching_keywords,
        # grammar_check,
    ]

    docs, _, codes = load_code_doc(data_file)
    if level == "corpus":
        for k, v in eval_items.items():
            print("\n" + k)
            if not isinstance(v, str):
                globals().update(v)
            gen_docs = read_files(to_eval=v)
            assert len(gen_docs) == len(docs)
            eval_multi_metrics(
                gen_docs, docs, codes, eval_metrics, normalize=True, level=level
            )
            # compare with codes
            # codebertscore(gen_docs, codes, normalize=True)
    elif level == "sentence":
        eval_metrics = [
            bertscore,
            codebertscore,
            rouge_score,
            nltk_bleu,
            meteor_score,
            fact_scores,
            # branching_keywords,
            # grammar_check,
        ]

        df = pd.DataFrame(human_eval)
        for k, v in eval_items.items():
            print(k)
            globals().update(v)
            gen_docs = read_files()
            assert len(gen_docs) == len(docs)
            res = eval_multi_metrics(
                gen_docs, docs, codes, eval_metrics, normalize=False, level=level
            )
        if save:
            df.to_csv(os.path.join(current_dir, "sentence_eval.csv"), index=True)
    else:
        raise ValueError("level must be either corpus or sentence")


# %%
def compute_human_ranks(scores: Tuple[float, float], threshold=1e-6) -> Tuple[int, int]:
    """Given a pair of human direct assessment(DA) scores,
       computes the relative ranking. If the difference between
       the two scores is less than the provided threshold,
       the rank is the same.
    Args:
        scores (Tuple[int, int]): A tuple containing the 2 DA scores.
        threshold (int, optional): The threshold of the difference between two scores at which the
                                   the difference is considered (significant). Defaults to 25.
    Returns:
        Tuple[int, int]: The relative ranking of the provided scores
    """
    assert len(scores) == 2
    a, b = scores

    if (a == b) or abs(a - b) < threshold:
        return [1, 1]

    if a > b:
        return [1, 2]

    return [2, 1]


def get_index_pairs(df):
    pairs = list(itertools.combinations(df.index, 2))
    return [pair for pair in pairs if pair[0] != pair[1]]


# %%
comparison_variant_b = [
    (operator.lt, operator.lt, "c"),
    (operator.lt, operator.eq, "t"),
    (operator.lt, operator.gt, "d"),
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "t"),
    (operator.gt, operator.gt, "c"),
]

comparison_variant_c = [
    (operator.lt, operator.lt, "c"),
    (operator.lt, operator.eq, "d"),
    (operator.lt, operator.gt, "d"),
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "d"),
    (operator.gt, operator.gt, "c"),
]

comparison_variant_d = [
    (operator.lt, operator.lt, "c"),  # <, <
    (operator.lt, operator.eq, "t"),  # <, =
    (operator.lt, operator.gt, "d"),  # <, >
    (operator.eq, operator.lt, "t"),  # =, <
    (operator.eq, operator.eq, "c"),
    (operator.eq, operator.gt, "t"),  # =, >
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "t"),
    (operator.gt, operator.gt, "c"),
]


def compute_rank_pair_type(
    human_ranking: Iterable[Union[int, float]],
    metric_ranking: Iterable[Union[int, float]],
) -> Union[Literal["c"], Literal["d"], Literal["t"], Literal["-"]]:
    comparison_table = comparison_variant_b

    for h_op, m_op, outcome in comparison_table:
        if h_op(*human_ranking) and m_op(*metric_ranking):
            return outcome

    return "-"


# %% compute kendalls_tau
def kendalls_tau(
    df: pd.DataFrame, human_col: str, metric_col: str, threshold: Number = 0.0001
) -> Tuple[float, Number, Number, Number]:
    counts = Counter()

    pairs = get_index_pairs(df)
    pair_types = []

    for pair in pairs:
        pair_df = df[df.index.isin(pair)]
        human_scores = pair_df[human_col]
        metric_scores = pair_df[metric_col]

        human_ranks = compute_human_ranks(human_scores, threshold=threshold)
        metric_ranks = metric_scores.rank(method="max", ascending=False)

        pair_type = compute_rank_pair_type(human_ranks, metric_ranks)
        pair_types.append(pair_type)
    counts.update(pair_types)
    concordant_pairs = counts["c"]
    discordant_pairs = counts["d"]
    ties = counts["t"]
    tau = (concordant_pairs - discordant_pairs) / (
        concordant_pairs + discordant_pairs + ties
    )
    print(f"{metric_col} tau: {tau}")
    return tau, concordant_pairs, discordant_pairs, ties


# %% compare pipeline
def compare_pipeline(
    save: bool = False, level: Literal["corpus", "sentence"] = "corpus"
) -> Dict:
    docs, _, codes = load_code_doc()
    eval_items = {
        "codex": {"CODEX": True, "BY_CHUNKS": False},
        "codex(by_chunk)": {"CODEX": True, "BY_CHUNKS": True},
        "gpt-neo": {"CODEX": False, "BY_CHUNKS": False, "FINE_TUNED": False},
        "gpt-neo(by_chunk)": {"CODEX": False, "BY_CHUNKS": True, "FINE_TUNED": False},
        "gpt-neo(fine_tuned)": {"CODEX": False, "BY_CHUNKS": False, "FINE_TUNED": True},
        "gpt-neo(fine_tuned)(by_chunk)": {
            "CODEX": False,
            "BY_CHUNKS": True,
            "FINE_TUNED": True,
        },
    }
    eval_metrics = [
        bertscore,
        codebertscore,
        rouge_score,
        nltk_bleu,
        meteor_score,
        fact_scores,
        logicscore,
        # branching_keywords,
        # grammar_check,
    ]
    taus = {}
    for metric in eval_metrics:
        taus[metric.__name__] = {
            "concordant_pairs": 0,
            "discordant_pairs": 0,
            "ties": 0,
        }
    df = pd.DataFrame(human_eval)
    records = {"human_eval": []}  # torecord all results
    for metric in eval_metrics:
        records[metric.__name__] = []
    for trial, settings in eval_items.items():  # each model's result to compute
        print(trial)
        globals().update(settings)
        gen_docs = read_files()
        # first compute the dataframe containing the scores for each example
        scores = eval_multi_metrics(gen_docs, docs, codes, eval_metrics, level=level)
        taus[trial] = {}
        records["human_eval"].extend(df["human_eval_" + trial].tolist())
        for k, s in scores.items():  # k is the metric name
            df[k] = s[:19]  # FIXME: 19 is the hardcoded number of examples
            # then compute the tau for this model's result and all eval metrics
            tau, concordant, discordant, ties = kendalls_tau(
                df, human_col="human_eval_" + trial, metric_col=k
            )

            taus[trial][k] = tau
            taus[k]["concordant_pairs"] += concordant
            taus[k]["discordant_pairs"] += discordant
            taus[k]["ties"] += ties

            records[k].extend(df[k].tolist())

    # overall tau
    for metric in eval_metrics:
        taus[metric.__name__]["overall _tau"] = (
            taus[metric.__name__]["concordant_pairs"]
            - taus[metric.__name__]["discordant_pairs"]
        ) / (
            taus[metric.__name__]["concordant_pairs"]
            + taus[metric.__name__]["discordant_pairs"]
            + taus[metric.__name__]["ties"]
        )

    record_df = pd.DataFrame(records)
    if save:
        record_df.to_csv("records.csv")

    # compute the overall tau using cobined score
    record_df["combined"] = (
        4.097 * record_df["bertscore"] + 3.964 * record_df["rouge_score"]
    )
    tau, concordant, discordant, ties = kendalls_tau(
        record_df, human_col="human_eval", metric_col="combined"
    )

    # intergrate to the overall tau across models
    # tau_overall = func_to_compute_overall_tau
    return taus, record_df


# %% save records and select metrics
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def lasso_selction(df: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    # # FIXME: hack to normalize the scale of scores
    # df["bertscore"] = df["bertscore"] - 0.75
    # df["codebertscore"] = df["codebertscore"] - 0.65
    # df["nltk_bleu"] = df["nltk_bleu"] * 10

    X = df.drop(columns=["human_eval"])
    y = df["human_eval"]

    # reg = LassoCV(cv=5, alphas=np.logspace(-2.8, 1, 5), random_state=42)
    reg = LassoCV(cv=5, positive=True, random_state=42)
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X, y))
    coef = pd.Series(reg.coef_, index=X.columns)
    print(
        "Lasso picked "
        + str(sum(coef != 0))
        + " variables and eliminated the other "
        + str(sum(coef == 0))
        + " variables"
    )

    imp_coef = coef.sort_values()
    matplotlib.rcParams["figure.figsize"] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")
    return imp_coef


# df = pd.DataFrame(
#     {
#         "generated_description": gen_docs,
#         "reference_description": docs,
#         "code": codes,
#         # "code blocks": ["\n\n".join(d["blocks_codes"]) for d in data],
#     }
# )

# # %%
# with open(
#     os.path.join(
#         current_dir, f"generated-finetune_{FINE_TUNED}-by_chunks_{BY_CHUNKS}.csv"
#     ),
#     "w",
# ) as f:
#     df.to_csv(f)

# %% [markdown]
# # Actual run
# %%
if __name__ == "__main__":
    # for ml metrics
    # model_path = (
    #     "/home/v-haotiancui/NL2Code/Copilot-2/code2text/eval/results/checkpoint-best"
    # )
    # tokenizer = BertTokenizerFast.from_pretrained(model_path)
    # model = BertForSequenceClassification.from_pretrained(model_path)
    if args.mode == "generate":
        print("generating on test set using gpt-neo...")
        generate_gpt_neo_pipeline(save=args.save)
    elif args.mode == "eval":
        print("evaluating on test set...")
        eval_pipeline(save=args.save, level=args.level)
    elif args.mode == "both":
        print("1. generating on test set using gpt-neo...")
        generate_gpt_neo_pipeline(save=args.save)
        print("2. evaluating on test set...")
        eval_pipeline(save=args.save, level=args.level)
    elif args.mode == "compare":
        print("compare to human evaluation using kindall's tau...")
        assert args.level == "sentence", "only sentence level is supported for compare"
        taus, records = compare_pipeline(save=args.save, level=args.level)
        print(taus)
    else:
        raise ValueError(
            f"invalid mode. Please use a mode in generate, eval or both"
            "and type in python {__file__} mode"
        )
# %%
