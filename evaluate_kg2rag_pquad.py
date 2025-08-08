import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import eval, parse_result

with open(r"data/pquad/PQuad_json.json", "r", encoding="utf-8") as f:
    gold = json.load(f)

# pathes = list(Path(r"G:\jupyter\pycharm\KG2RAG\output\pquad").glob('*.json'))
pathes = list(Path("output/KG2RAG/pquad").glob('*.json'))


def extract_number(filepath):
    match = re.search(r'_(\d+)\.json$', filepath.name)
    return int(match.group(1)) if match else 0


sorted_pathes = sorted(pathes, key=extract_number)
numbers = []
numbers_gemma = []
em_scores = []
em_scores_gemma = []
for p in sorted_pathes:
    print(p.stem, "Top k:", p.stem.split("_")[-1])
    with open(p, "r", encoding="utf-8") as f:
        prediction = json.load(f)
    metrics = eval(prediction, "PQuad_json.json")
    correct = 0
    for dp in gold:
        id_ = dp['id']
        id_ = str(id_)
        gold_data = dp['answer']
        predicted_data = prediction['answer'][id_]

        if parse_result(gold_data, predicted_data):
            correct += 1
    print("Acc:", correct / len(gold))
    print()
    print("-" * 50)
    match = re.search(r'_(\d+)\.json$', p.name)
    num = int(match.group(1))
    if "gemini" in str(p):
        numbers.append(num)
        em_scores.append(metrics['em'])
    else:
        numbers_gemma.append(num)
        em_scores_gemma.append(metrics['em'])

test_df = pd.read_csv("test_df_pquad.csv", encoding="utf-8")

ls = list(Path("./results/pquad/").glob("*.pickle"))


def extract_number(filepath):
    match = re.search(r'_(\d+)\.pickle$', filepath.name)
    return int(match.group(1)) if match else 0


sorted_ls = sorted(ls, key=extract_number)

numbers_semrag = []
em_scores_semrag = []
for p in sorted_ls:
    with open(p, "rb") as f:
        responses = pickle.load(f)
    predictions = {"answer": {}, "sp": {}}
    for i, row in test_df.iterrows():
        predictions["answer"][str(row["id"])] = responses[i]["response"]
        sp_facts = []
        for cont in responses[i]["context"]:
            sp_facts.append([cont.metadata["title"], cont.metadata["index"]])
        predictions["sp"][str(row["id"])] = sp_facts
    print(p)
    res = eval(predictions, "PQuad_json.json")
    print()
    match = re.search(r'_(\d+)\.pickle$', p.name)
    num = int(match.group(1))

    numbers_semrag.append(num)
    em_scores_semrag.append(res['em'])

plt.figure(figsize=(6, 5))
plt.plot(numbers, em_scores, marker='x', linewidth=2, markersize=8, color='tab:orange',
         label='KG2RAG (gemini-2.5-flash)')
plt.plot(numbers_gemma, em_scores_gemma, marker='o', linewidth=2, markersize=8, color='tab:blue', label='KG2RAG (gemma3-27B)')
plt.plot(numbers_semrag, em_scores_semrag, marker='^', linestyle='--', linewidth=2, markersize=8, color='tab:olive',
         label='Semantic RAG')
plt.xlabel('k')
plt.ylabel('Exact Match (EM) Score')
plt.title('EM Score vs k - PQuAD Dataset')
# plt.grid(True, alpha=0.3)
plt.xticks(numbers_semrag)  # Show all numbers on x-axis

# Add value labels on points
# for i, (x, y) in enumerate(zip(numbers, em_scores)):
#     plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
# for i, (x, y) in enumerate(zip(numbers_gemini, em_scores_gemini)):
#     plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("pquad_final.jpg")
plt.show()
