import os
import sys
import re
from pathlib import Path
import pandas as pd
import jsonlines
from datasets import load_dataset
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

class ProcessACCA():
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _pase_answer(self, answer: str):

        if isinstance(answer, float):
            res_str = str(answer)
        elif isinstance(answer, int):
            res_str = str(answer)
        elif isinstance(answer, str):
            answer = answer.strip()
            pattern = r"([A-Z])[\.\)]\s(.+?)(?=\n[A-Z][\.\)]|\Z)"
            matches = re.findall(pattern, answer, re.DOTALL)
            options = {key: value.strip() for key, value in matches}

            option_keys = list(sorted(list(options.keys())))

            res_str = ",".join(option_keys)
        else:
            res_str = ""

        return res_str

    def _parse_problem(self, problem: str):
        problem = problem.strip()

        replace_dict = {"  ": "", " ": ""}
        pattern = re.compile("|".join(re.escape(k) for k in replace_dict.keys()))
        problem = pattern.sub(lambda x: replace_dict[x.group()], problem)
        return problem

    def parse_data(self):
        datas = []

        bt_path = os.path.join(self.data_path, "BT_QA.xlsx")
        df = pd.read_excel(bt_path, sheet_name="Sheet1")
        df = df.rename(columns={"Question": "problem", "Answer": "answer"})
        df = df[["problem", "answer"]]
        df["problem"] = df["problem"].apply(lambda x: self._parse_problem(x))
        df["answer"] = df["answer"].apply(lambda x: self._pase_answer(x))
        for index, row in df.iterrows():
            item = {
                "problem": row["problem"],
                "answer": row["answer"]
            }
            datas.append(item)

        ma_path = os.path.join(self.data_path, "MA_QA.xlsx")
        df = pd.read_excel(ma_path, sheet_name="Sheet1")
        df = df.rename(columns={"Question": "problem", "Answer": "answer"})
        df = df[["problem", "answer"]]
        df["problem"] = df["problem"].apply(lambda x: self._parse_problem(x))
        df["answer"] = df["answer"].apply(lambda x: self._pase_answer(x))
        for index, row in df.iterrows():
            item = {
                "problem": row["problem"],
                "answer": row["answer"]
            }
            datas.append(item)
        return datas


if __name__ == "__main__":

    data_path = os.path.join(root, "datasets/raw/ACCA")
    process = ProcessACCA(data_path)
    datas = process.parse_data()

    with jsonlines.open(os.path.join(data_path, "data.jsonl"), "w") as writer:
        for data in datas:
            writer.write(data)

    train_data, test_data = train_test_split(datas, test_size=0.2, random_state=42)

    train_path = os.path.join(data_path, "train.jsonl")
    with jsonlines.open(train_path, "w") as writer:
        for data in train_data:
            writer.write(data)

    test_path = os.path.join(data_path, "test.jsonl")
    with jsonlines.open(test_path, "w") as writer:
        for data in test_data:
            writer.write(data)

    dataset = DatasetDict({
        "data": load_dataset("json", data_files=os.path.join(data_path, "data.jsonl"), split="train"),
        "train": load_dataset("json", data_files=train_path, split="train"),
        "test": load_dataset("json", data_files=test_path, split="train")
    })

    dataset.push_to_hub("zwt963/FinReasoner")