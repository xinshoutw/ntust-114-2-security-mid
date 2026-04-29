"""讀取 UCI Adult 資料、清洗、裁減欄位、輸出 original_clean.csv。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HW1_DATA_PATH = Path("/Users/xinshou/IdeaProjects/ntust-114-2-security-hw1/data/adult.data")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "original"
OUTPUT_PATH = OUTPUT_DIR / "original_clean.csv"

DATA_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

# 5 個數值準識別碼，與 HW1 一致
NUMERIC_QI = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
# 4 個保留的類別欄位
CATEGORICAL_KEEP = ["workclass", "education", "marital-status", "sex"]
SENSITIVE_ATTR = "income"

KEEP_COLUMNS = NUMERIC_QI + CATEGORICAL_KEEP + [SENSITIVE_ATTR]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(HW1_DATA_PATH, header=None, names=DATA_COLUMNS, skipinitialspace=True)
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    df = df.drop(columns=["fnlwgt"])
    df[SENSITIVE_ATTR] = (df[SENSITIVE_ATTR].str.strip().str.rstrip(".") == ">50K").astype(int)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    df = df[KEEP_COLUMNS].copy()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"rows={len(df)}, cols={df.shape[1]}")
    print(f"columns: {list(df.columns)}")
    print(f"dtypes: {dict(df.dtypes.astype(str))}")
    print(f"missing per col: {df.isna().sum().to_dict()}")
    print(f"income dist: {df[SENSITIVE_ATTR].value_counts().to_dict()}")
    print(df[NUMERIC_QI].describe().round(2))


if __name__ == "__main__":
    main()
