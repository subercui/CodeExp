# %% [markdown]
# # Process human eval result files
# This script reads the files and merge eval scores into one csv. The result csv
# contains scores for each model and each sample.
# 1. count human-eval scores for every model, including overall score and each metric.
# Save the results to jsonl and csv files. Each row are scores for one example.
# 2. Correlation between style score and other metrics.
# 3. pick some best examples.
# %%
import os
import json
import pandas as pd
import numpy as np
import argparse
import pprint
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Callable, List, Dict

from utils import (
    Evaluator,
    fast_kendalls_tau,
    isnotebook,
    kendalls_tau,
    rouge_1f,
    rouge_lf,
    rouge_score,
    nltk_bleu,
    meteor_score,
    bertscore,
    codebertscore,
    fact_scores,
)

sns.set_theme(context="notebook", font_scale=1.2)
# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data-dir",
    type=str,
    required=False,
    default="/home/v-haotiancui/NL2Code/Copilot-2/dataset/human-eval/results",
    help="Path to the data directory human eval results.",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=False,
    default="/home/v-haotiancui/NL2Code/Copilot-2/dataset/human-eval/",
    help="Directory to save the processed human eval results, in jsonl and csv format.",
)
parser.add_argument(
    "--internal-dir",
    type=str,
    required=False,
    default="/home/v-haotiancui/NL2Code/Copilot-2/dataset/human-eval/internal",
    help="Directory containing the internal mapping files. Those files have the"
    "testset_ids to mapp human-eval examples to test set",
)
parser.add_argument(
    "--replace",
    action="store_true",
    default=False,
    help="Replace the saved file with newly generated file",
)

# list of models used in make_human_eval_set.py. The order is kept.
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

if isnotebook():
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

# %%
# make dataframe by metric steps
def _to_gens_df(results: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of results to a dataframe of generated texts of all models.
    """
    gens_df = pd.DataFrame(
        columns=[
            "human_eval_num",
            "testset_id",
            "code",
            "reference",
            "model",
            "gen",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "avg_s1-4",
            "avg_s1-5",
        ]
    )
    for res in results:
        human_eval_num = res["human_eval_num"]
        testset_id = res["testset_id"]
        code = res["code"]
        reference = res["reference"]["docstring"]
        for m in models:
            if m != "reference":
                gen = res[m]["docstring"]
                s1, s2, s3, s4, s5 = res[m]["step 1-5 scores"]
                avg_s1_4 = res[m]["avg step 1-4 scores"]
                avg_s1_5 = res[m]["avg step 1-5 scores"]
                gens_df.loc[len(gens_df)] = [
                    human_eval_num,
                    testset_id,
                    code,
                    reference,
                    m,
                    gen,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    avg_s1_4,
                    avg_s1_5,
                ]
    return gens_df


def res2csvrow(res: Dict) -> Dict:
    """
    Convert a single result to a row in csv file.
    """
    row = {}
    row["human_eval_num"] = res["human_eval_num"]
    row["testset_id"] = res["testset_id"]
    row["code"] = res["code"]
    for m in models:
        row[f"{m}_docstring"] = res[m]["docstring"]
        row[f"{m}_s1"] = res[m]["step 1-5 scores"][0]
        row[f"{m}_s2"] = res[m]["step 1-5 scores"][1]
        row[f"{m}_s3"] = res[m]["step 1-5 scores"][2]
        row[f"{m}_s4"] = res[m]["step 1-5 scores"][3]
        row[f"{m}_s5"] = res[m]["step 1-5 scores"][4]
        row[f"{m}_avg_s1-4"] = res[m]["avg step 1-4 scores"]
        row[f"{m}_avg_s1-5"] = res[m]["avg step 1-5 scores"]
    return row


# score stats for one model
def model_stats(results: List[Dict], model_name: str) -> Dict:
    """
    Compute the stats for one model. including mean and std.
    """
    stats = {}
    all_scores = {
        "s1": [],
        "s2": [],
        "s3": [],
        "s4": [],
        "s5": [],
        "avg_s1-4": [],
        "avg_s1-5": [],
    }
    for res in results:
        model_res = res[model_name]
        all_scores["s1"].append(model_res["step 1-5 scores"][0])
        all_scores["s2"].append(model_res["step 1-5 scores"][1])
        all_scores["s3"].append(model_res["step 1-5 scores"][2])
        all_scores["s4"].append(model_res["step 1-5 scores"][3])
        all_scores["s5"].append(model_res["step 1-5 scores"][4])
        all_scores["avg_s1-4"].append(model_res["avg step 1-4 scores"])
        all_scores["avg_s1-5"].append(model_res["avg step 1-5 scores"])
    for key in all_scores.keys():
        stats[key] = {
            "mean": np.mean(all_scores[key]),
            "std": np.std(all_scores[key]),
        }
    print(f"{model_name}:")
    # format dict for printing, float to 3 decimal places
    pprint.pprint(
        {
            k: {"mean": round(v["mean"], 3), "std": round(v["std"], 4)}
            for k, v in stats.items()
        }
    )
    return stats, all_scores


# %% [markdown]
# # Read human eval results
files = os.listdir(args.data_dir)
data = {}
for file in files:
    try:
        data[int(file.split(".")[0])] = pd.read_csv(os.path.join(args.data_dir, file))
        assert "code" in data[int(file.split(".")[0])].columns
    except:
        print(file)

# %%
all_results = []
for k, v in data.items():
    # k is the file number
    example_results = {}
    example_results["human_eval_num"] = int(k)
    _tmp = pd.read_csv(os.path.join(args.internal_dir, f"{k}.csv"))
    example_results["testset_id"] = int(_tmp["testset_id"].values[0])
    example_results["code"] = _tmp["code"].values[0]
    for i, m in enumerate(models):
        example_results[m] = {}
        example_results[m]["docstring"] = v["docstring"].values[i]
        example_results[m]["step 1-5 scores"] = (
            int(v["step 1. explaining"].values[i]),
            int(v["step 2. informative/coverage"].values[i]),
            int(v["step 3. coherence/correctness"].values[i]),
            int(v["step 4. readability/fluency"].values[i]),
            int(v["step 5. format/style"].values[i]),
        )
        example_results[m]["avg step 1-4 scores"] = np.mean(
            example_results[m]["step 1-5 scores"][:4]
        )
        example_results[m]["avg step 1-5 scores"] = np.mean(
            example_results[m]["step 1-5 scores"]
        )
    # make sure the reference is the original
    example_results["reference"]["docstring"] = _tmp[_tmp["model"] == "reference"][
        "docstring"
    ].values[0]

    all_results.append(example_results)
all_results.sort(key=lambda x: x["human_eval_num"])

# %% [markdown]
# ## Save human eval results
output_jsonl = os.path.join(args.save_dir, "human_eval_results.jsonl")
output_csv = os.path.join(args.save_dir, "human_eval_results.csv")
if args.replace or not os.path.exists(output_jsonl):
    with open(output_jsonl, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

all_gens_df = _to_gens_df(all_results)
all_results_df = pd.DataFrame([res2csvrow(res) for res in all_results])
if args.replace or not os.path.exists(output_csv):
    all_results_df.to_csv(output_csv, index="human_eval_num")
# %% [markdown]
# # Results stats
all_stats = {}
all_scores = {}
for m in models:
    all_stats[m], all_scores[m] = model_stats(all_results, m)

# %% [markdown]
# ## Histograms (w/ kde) and violin plots

# make dataframe of avg step1-4 scores for all models
plot_df = pd.DataFrame({m: all_scores[m]["avg_s1-4"] for m in models})

# violinplot of avg step 1-4 scores for all models
fig, ax = plt.subplots(figsize=(10, 10))
sns.violinplot(
    data=plot_df,
    ax=ax,
    bw=0.15,
    cut=0,
)
ax.set_title("Average step 1-4 scores for all models")
ax.set_xlabel("Model")
ax.set_ylabel("Average step 1-4 scores")
ax.set_xticklabels(models)
# plt.savefig(os.path.join(args.save_dir, "avg_step1-4_violin.png"))

# %%
# kdeplot of avg step 1-4 scores for all models
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
model_kde_kws = {
    "reference": {"linewidth": 2, "alpha": 0.6, "label": "Reference"},
    "ft12_codeT5": {"linewidth": 3, "label": "codeT5-(r+r)"},
    "ft12_gpt-neo-27": {
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.3,
        "label": "GPT-Neo27-(r+r)",
    },
    "ft12_gpt-neo": {
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.3,
        "label": "GPT-Neo13-(r+r)",
    },
    "ft12_gpt2": {
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.3,
        "label": "GPT-2-base-(r+r)",
    },
    "ft1_codeT5": {
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.3,
        "label": "CodeT5-(raw)",
    },
    "ft2_codeT5": {"linewidth": 3, "label": "CodeT5-(refined)"},
    "codex_Py2Doc": {"linewidth": 2, "label": "Codex-Py2Doc"},
    "codex_Py2NL": {"linewidth": 2, "label": "Codex-Py2NL"},
}
for m in models:
    sns.kdeplot(
        all_scores[m]["avg_s1-4"],
        ax=ax,
        cut=0,  # clip=(0, 4), fill=True,
        bw_adjust=0.7,
        **model_kde_kws.get(m, {}),
    )
ax.set_xlabel("Overall scores")
# ax.set_xlim(0, 4)
ax.set_ylabel("Density")
ax.set_title("Distribution of overall scores for all human-evaluated models, w/ KDE")
ax.annotate(
    "",
    xy=(0.78, 0.84),
    xycoords="axes fraction",
    xytext=(-42, 18),
    textcoords="offset points",
    size=20,
    arrowprops=dict(arrowstyle="fancy", fc="0.6", ec="none"),
)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "avg_step_1-4_scores_kde.png"))
plt.show()

# %% [markdown]
# ## Kendall's Tau between auto metrics and human evaluations
# 4+1 human eval aspects and 6+1 auto metrics, that should generate a 5x6 table
# of scores.

# %% make the dataframe wheree each coloumn is a metric
# update all_results with auto metric scores
auto_metrics: List[Callable] = [
    rouge_1f,
    rouge_lf,
    nltk_bleu,
    meteor_score,
    bertscore,
    codebertscore,
    fact_scores,
]
for sent_metric in auto_metrics:
    Evaluator(sent_metric).eval_df(
        all_gens_df,
        code_col="code",
        ref_col="reference",
        gen_col="gen",
        out_col=sent_metric.__name__,
    )

# %% using code as the reference
for sent_metric in auto_metrics:
    if sent_metric is not fact_scores:
        Evaluator(sent_metric).eval_df(
            all_gens_df,
            code_col="code",
            ref_col="code",
            gen_col="gen",
            out_col=sent_metric.__name__ + "_to_code",
        )

# %% save all gens dataframe to csv
if args.replace or not os.path.exists(os.path.join(args.save_dir, "all_scores.csv")):
    all_gens_df.to_csv(os.path.join(args.save_dir, "all_scores.csv"), index=False)

# %% compute kendalls tau
human_aspect_cols = ["s1", "s2", "s3", "s4", "avg_s1-4"]
auto_metrics_cols = [m.__name__ for m in auto_metrics] + [
    m.__name__ + "_to_code" for m in auto_metrics if m is not fact_scores
]
tau_df = pd.DataFrame(columns=auto_metrics_cols)
info_df = pd.DataFrame(columns=auto_metrics_cols)
for h_col in human_aspect_cols:
    print(f"computing kendall's tau for {h_col} ...")
    tau_row = []
    info_row = []
    for i, a_col in enumerate(auto_metrics_cols):
        print(f"\t{i+1}/{len(auto_metrics_cols)}: {a_col}")
        tau, concordant, discordant, ties = fast_kendalls_tau(
            all_gens_df,
            human_col=h_col,
            metric_col=a_col,
        )
        tau_row.append(tau)
        info_row.append((concordant, discordant, ties))
    tau_df.loc[h_col] = tau_row
    info_df.loc[h_col] = info_row

# %% save kendall's tau dataframe to csv
if args.replace or not os.path.exists(os.path.join(args.save_dir, "kendalls_tau.csv")):
    tau_df.to_csv(os.path.join(args.save_dir, "kendalls_tau.csv"), index=True)

# %% [markdown]
# # Pick best examples

print("best CodeT5-(m) examples:")
all_results_df[all_results_df["ft2_codeT5_avg_s1-4"] == 4.0][
    all_results_df["codex_Py2Doc_avg_s1-4"] < 3.0
]
# %%
print("best CodeT5-(l,m) examples:")
all_results_df[all_results_df["ft12_codeT5_avg_s1-4"] == 4.0][
    all_results_df["codex_Py2Doc_avg_s1-4"] < 3.0
]

# %%
