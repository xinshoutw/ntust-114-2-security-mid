"""跑 PrivBayes (DataSynthesizer 0.1.13) 對 5 個 ε × 3 個 seed 各做一次。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "original" / "original_clean.csv"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
DESC_DIR = PROJECT_ROOT / "results" / "metrics"
TIMING_PATH = PROJECT_ROOT / "results" / "metrics" / "privbayes_timing.json"

EPSILONS = [0.1, 0.5, 1.0, 2.0, 4.0]
SEEDS = [42, 123, 456]
DEGREE = 2  # k = max parents in Bayesian network

ATTRIBUTE_IS_CATEGORICAL = {
    "workclass": True,
    "education": True,
    "marital-status": True,
    "sex": True,
    "income": True,
    "age": False,
    "education-num": False,
    "hours-per-week": False,
    "capital-gain": False,
    "capital-loss": False,
}
CATEGORY_THRESHOLD = 20


def run_one(epsilon: float, seed: int, n_tuples: int) -> dict:
    tag = f"eps{epsilon}_seed{seed}"
    desc_path = DESC_DIR / f"description_{tag}.json"
    synth_path = SYNTH_DIR / f"synthetic_{tag}.csv"

    print(f"  [eps={epsilon}, seed={seed}] ", end="", flush=True)

    t0 = time.perf_counter()
    describer = DataDescriber(category_threshold=CATEGORY_THRESHOLD)
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=str(INPUT_PATH),
        epsilon=epsilon,
        k=DEGREE,
        attribute_to_is_categorical=ATTRIBUTE_IS_CATEGORICAL,
        seed=seed,
    )
    describer.save_dataset_description_to_file(str(desc_path))
    t_desc = time.perf_counter() - t0

    t1 = time.perf_counter()
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(n=n_tuples, description_file=str(desc_path))
    generator.save_synthetic_data(str(synth_path))
    t_gen = time.perf_counter() - t1

    print(f"describer={t_desc:.1f}s, generator={t_gen:.1f}s")
    return {
        "epsilon": epsilon,
        "seed": seed,
        "time_describe_sec": round(t_desc, 3),
        "time_generate_sec": round(t_gen, 3),
        "time_total_sec": round(t_desc + t_gen, 3),
        "synth_path": str(synth_path.relative_to(PROJECT_ROOT)),
        "desc_path": str(desc_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    DESC_DIR.mkdir(parents=True, exist_ok=True)

    orig_df = pd.read_csv(INPUT_PATH)
    n_tuples = len(orig_df)
    print(f"input shape={orig_df.shape}, eps={EPSILONS}, seeds={SEEDS}, k={DEGREE}\n")

    timing = []
    t_start = time.perf_counter()
    for eps in EPSILONS:
        for seed in SEEDS:
            timing.append(run_one(eps, seed, n_tuples))
    t_total = time.perf_counter() - t_start

    TIMING_PATH.write_text(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"\nDone. total elapsed: {t_total:.1f}s")
    print(pd.DataFrame(timing).pivot_table(index="epsilon", columns="seed", values="time_total_sec").to_string())


if __name__ == "__main__":
    main()
