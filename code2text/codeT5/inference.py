# %%
import json
import numpy as np
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# %% inference fo codeT5_base multi summary model

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base-multi-sum")

text = """def svg_to_image(string, size=None):
if isinstance(string, unicode):
    string = string.encode('utf-8')
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
if not renderer.isValid():
    raise ValueError('Invalid SVG data.')
if size is None:
    size = renderer.defaultSize()
    image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
return image"""

input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=20)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints: "Convert a SVG string to a QImage."

# %% check the length of texts in dataset after tokenization
data_file = (
    "/home/v-haotiancui/NL2Code/Copilot-2/code2text/codeT5/"
    "data/GithubCodeDocStep1n2Filtered.train.jsonl"
)
with open(data_file, "r") as f:
    data = [json.loads(line) for line in f]

codes = [d["code"] for d in data]
docstrings = [d["docstring"] for d in data]
combined_texts = [d["code"] + " " + d["docstring"] for d in data]

# %%
code_tokens = tokenizer(codes, truncation=False)
# the length of all codes
num_tokens = []
for i, code in enumerate(codes):
    num_tokens.append(len(code_tokens["input_ids"][i]))
print(f"max length of code: {max(num_tokens)}")
print(f"min length of code: {min(num_tokens)}")
print(f"avg length of code: {sum(num_tokens) / len(num_tokens)}")
num_tokens = np.array(num_tokens)
print(
    f"num, percent of codes with length 512: "
    f"{(num_tokens <= 512).sum() / len(num_tokens) * 100:.4f}%"
)
print(
    f"num, percent of codes with length 1024: "
    f"{(num_tokens <= 1024).sum() / len(num_tokens) * 100:.4f}%"
)
print(
    f"num, percent of codes with length 2048: "
    f"{(num_tokens <= 2048).sum() / len(num_tokens) * 100:.4f}%"
)
"""
max length of code: 283871
min length of code: 39
avg length of code: 245.8436503315952
num, percent of codes with length 512: 95.7620%
num, percent of codes with length 1024: 99.8545%
num, percent of codes with length 2048: 99.9772%
"""

# %%
docstring_tokens = tokenizer(docstrings, truncation=False)
# the length of all docstrings
num_tokens = []
for i, docstring in enumerate(docstrings):
    num_tokens.append(len(docstring_tokens["input_ids"][i]))
print(f"max length of docstring: {max(num_tokens)}")
print(f"min length of docstring: {min(num_tokens)}")
print(f"avg length of docstring: {sum(num_tokens) / len(num_tokens)}")
num_tokens = np.array(num_tokens)
print(
    f"num, percent of docstrings with length 512: "
    f"{(num_tokens <= 512).sum() / len(num_tokens) * 100:.4f}%"
)
print(
    f"num, percent of docstrings with length 1024: "
    f"{(num_tokens <= 1024).sum() / len(num_tokens) * 100:.4f}%"
)
print(
    f"num, percent of docstrings with length 2048: "
    f"{(num_tokens <= 2048).sum() / len(num_tokens) * 100:.4f}%"
)
"""
max length of docstring: 9326
min length of docstring: 6
avg length of docstring: 161.3275388548575
num, percent of docstrings with length 512: 95.8411%
num, percent of docstrings with length 1024: 99.6077%
num, percent of docstrings with length 2048: 99.9753%
"""

# %%
combined_tokens = tokenizer(combined_texts, truncation=False)
# the length of all combined texts
num_tokens = []
for i, combined_text in enumerate(combined_texts):
    num_tokens.append(len(combined_tokens["input_ids"][i]))
print(f"max length of combined text: {max(num_tokens)}")
print(f"min length of combined text: {min(num_tokens)}")
print(f"avg length of combined text: {sum(num_tokens) / len(num_tokens)}")
num_tokens = np.array(num_tokens)
print(
    f"num, percent of combined texts with length 512: "
    f"{(num_tokens <= 512).sum() / len(num_tokens) * 100:.4f}%"
)
print(
    f"num, percent of combined texts with length 1024: "
    f"{(num_tokens <= 1024).sum() / len(num_tokens) * 100:.4f}%"
)
print(
    f"num, percent of combined texts with length 2048: "
    f"{(num_tokens <= 2048).sum() / len(num_tokens) * 100:.4f}%"
)
"""
max length of combined text: 283936
min length of combined text: 58
avg length of combined text: 405.2834632714018
num, percent of combined texts with length 512: 75.6252%
num, percent of combined texts with length 1024: 97.4972%
num, percent of combined texts with length 2048: 99.9095%
"""
# %%
