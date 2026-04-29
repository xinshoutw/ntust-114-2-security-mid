"""對每組 (eps, seed) 算 TVD / 相關係數差 / TSTR (LR + RF)，並聚合畫圖。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIG_PATH = PROJECT_ROOT / "data" / "original" / "original_clean.csv"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

EPSILONS = [0.1, 0.5, 1.0, 2.0, 4.0]
SEEDS = [42, 123, 456]
SPLIT_SEED = 42
NUM_BINS = 20
TEST_SIZE = 0.2
TARGET_COL = "income"

NUMERIC_COLS = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
CATEGORICAL_COLS = ["workclass", "education", "marital-status", "sex"]
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


def tvd_categorical(orig: pd.Series, synth: pd.Series) -> tuple[float, pd.DataFrame]:
    p_orig = orig.value_counts(normalize=True)
    p_synth = synth.value_counts(normalize=True)
    all_vals = sorted(set(p_orig.index) | set(p_synth.index), key=str)
    p_orig = p_orig.reindex(all_vals, fill_value=0.0)
    p_synth = p_synth.reindex(all_vals, fill_value=0.0)
    tvd = 0.5 * float(np.abs(p_orig.values - p_synth.values).sum())
    return tvd, pd.DataFrame({"value": all_vals, "original": p_orig.values, "synthetic": p_synth.values})


def tvd_numeric(orig: pd.Series, synth: pd.Series) -> tuple[float, pd.DataFrame]:
    lo = float(min(orig.min(), synth.min()))
    hi = float(max(orig.max(), synth.max()))
    if hi == lo:
        return 0.0, pd.DataFrame({"bin": ["all"], "original": [1.0], "synthetic": [1.0]})
    edges = np.linspace(lo, hi, NUM_BINS + 1)
    o_counts, _ = np.histogram(orig.values, bins=edges)
    s_counts, _ = np.histogram(synth.values, bins=edges)
    p_orig = o_counts / max(o_counts.sum(), 1)
    p_synth = s_counts / max(s_counts.sum(), 1)
    tvd = 0.5 * float(np.abs(p_orig - p_synth).sum())
    centers = 0.5 * (edges[:-1] + edges[1:])
    return tvd, pd.DataFrame({"bin_center": centers, "original": p_orig, "synthetic": p_synth})


def compute_tvd_all(orig_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for col in orig_df.columns:
        if col in CATEGORICAL_COLS or col == TARGET_COL:
            tvd, _ = tvd_categorical(orig_df[col].astype(str), synth_df[col].astype(str))
        else:
            tvd, _ = tvd_numeric(orig_df[col], synth_df[col])
        out[col] = tvd
    return out


def plot_marginals(orig_df: pd.DataFrame, synth_df: pd.DataFrame, epsilon: float, out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tvd_per_col = {}
    n_cols = len(orig_df.columns)
    n_per_row = 3
    n_rows = (n_cols + n_per_row - 1) // n_per_row
    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(5 * n_per_row, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(orig_df.columns):
        ax = axes[idx]
        if col in CATEGORICAL_COLS or col == TARGET_COL:
            tvd, dist_df = tvd_categorical(orig_df[col].astype(str), synth_df[col].astype(str))
            x = np.arange(len(dist_df))
            w = 0.4
            ax.bar(x - w / 2, dist_df["original"], width=w, label="original", color="#1f77b4")
            ax.bar(x + w / 2, dist_df["synthetic"], width=w, label="synthetic", color="#ff7f0e")
            ax.set_xticks(x)
            ax.set_xticklabels(dist_df["value"].astype(str), rotation=45, ha="right", fontsize=7)
        else:
            tvd, dist_df = tvd_numeric(orig_df[col], synth_df[col])
            ax.plot(dist_df["bin_center"], dist_df["original"], label="original", color="#1f77b4")
            ax.plot(dist_df["bin_center"], dist_df["synthetic"], label="synthetic", color="#ff7f0e")
        tvd_per_col[col] = tvd
        ax.set_title(f"{col}\nTVD = {tvd:.3f}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[n_cols:]:
        ax.set_visible(False)
    plt.suptitle(f"Marginal Distributions (eps = {epsilon}, seed = {SEEDS[0]})", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / f"marginals_eps{epsilon}.png", dpi=130, bbox_inches="tight")
    plt.close()
    return tvd_per_col


def encode_for_corr(df: pd.DataFrame, encoders: Optional[dict[str, LabelEncoder]] = None):
    out = df.copy()
    out_enc: dict[str, LabelEncoder] = {} if encoders is None else encoders
    for col in CATEGORICAL_COLS:
        if encoders is None:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            out_enc[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            out[col] = out[col].astype(str).map(
                lambda x: int(np.where(le.classes_ == x)[0][0]) if x in known else len(le.classes_)
            )
    return out.astype(float), out_enc


def correlation_frobenius(orig_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    o_enc, encs = encode_for_corr(orig_df)
    s_enc, _ = encode_for_corr(synth_df, encoders=encs)
    co = o_enc.corr().fillna(0.0)
    cs = s_enc.corr().fillna(0.0)
    return float(np.linalg.norm(co.values - cs.values, ord="fro"))


def plot_correlation(orig_df: pd.DataFrame, synth_df: pd.DataFrame, epsilon: float, out_dir: Path) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    o_enc, encs = encode_for_corr(orig_df)
    s_enc, _ = encode_for_corr(synth_df, encoders=encs)
    co = o_enc.corr().fillna(0.0)
    cs = s_enc.corr().fillna(0.0)
    diff = co.values - cs.values
    fro = float(np.linalg.norm(diff, ord="fro"))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.heatmap(co, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0], cbar=False, annot_kws={"size": 7})
    axes[0].set_title("Original Correlation")
    sns.heatmap(cs, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], cbar=False, annot_kws={"size": 7})
    axes[1].set_title(f"Synthetic Correlation (eps = {epsilon}, seed = {SEEDS[0]})")
    sns.heatmap(np.abs(diff), annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, ax=axes[2], cbar=True, annot_kws={"size": 7})
    axes[2].set_title(f"|Difference|  (Frobenius = {fro:.3f})")
    plt.suptitle(f"Correlation Matrices (eps = {epsilon})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"correlation_eps{epsilon}.png", dpi=130, bbox_inches="tight")
    plt.close()
    return fro


def encode_features(df: pd.DataFrame, encoders: Optional[dict[str, LabelEncoder]] = None):
    X_df = df[ALL_FEATURES].copy()
    out_enc: dict[str, LabelEncoder] = {} if encoders is None else encoders
    for col in CATEGORICAL_COLS:
        if encoders is None:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            out_enc[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            X_df[col] = X_df[col].astype(str).map(
                lambda x: int(np.where(le.classes_ == x)[0][0]) if x in known else len(le.classes_)
            )
    X = X_df.astype(float).values
    y = df[TARGET_COL].astype(int).values
    return X, y, out_enc


def fit_predict(model_name: str, X_train, y_train, X_test, scaler: Optional[StandardScaler] = None):
    if model_name == "LR":
        if scaler is None:
            scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=2000, random_state=SPLIT_SEED)
        clf.fit(Xtr, y_train)
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
    elif model_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SPLIT_SEED, n_jobs=-1)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)
    else:
        raise ValueError(model_name)
    return pred, prob, scaler


def tstr_trtr(orig_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    orig_train_df, orig_test_df = train_test_split(
        orig_df, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=orig_df[TARGET_COL]
    )
    X_train_real, y_train_real, encs = encode_features(orig_train_df)
    X_test_real, y_test_real, _ = encode_features(orig_test_df, encoders=encs)
    X_train_synth, y_train_synth, _ = encode_features(synth_df, encoders=encs)

    out = {}
    for m in ["LR", "RF"]:
        pred_r, prob_r, _ = fit_predict(m, X_train_real, y_train_real, X_test_real)
        out[f"TRTR_{m}_acc"] = round(float(accuracy_score(y_test_real, pred_r)), 4)
        out[f"TRTR_{m}_f1"] = round(float(f1_score(y_test_real, pred_r)), 4)
        out[f"TRTR_{m}_auc"] = round(float(roc_auc_score(y_test_real, prob_r)), 4)
        pred_s, prob_s, _ = fit_predict(m, X_train_synth, y_train_synth, X_test_real)
        out[f"TSTR_{m}_acc"] = round(float(accuracy_score(y_test_real, pred_s)), 4)
        out[f"TSTR_{m}_f1"] = round(float(f1_score(y_test_real, pred_s)), 4)
        out[f"TSTR_{m}_auc"] = round(float(roc_auc_score(y_test_real, prob_s)), 4)
    return out


def evaluate_one(orig_df: pd.DataFrame, epsilon: float, seed: int) -> dict:
    synth_path = SYNTH_DIR / f"synthetic_eps{epsilon}_seed{seed}.csv"
    synth_df = pd.read_csv(synth_path)[orig_df.columns.tolist()]
    tvd_per_col = compute_tvd_all(orig_df, synth_df)
    avg_tvd = float(np.mean(list(tvd_per_col.values())))
    fro = correlation_frobenius(orig_df, synth_df)
    ml = tstr_trtr(orig_df, synth_df)
    return {
        "epsilon": epsilon,
        "seed": seed,
        "avg_TVD": round(avg_tvd, 4),
        "corr_frobenius_diff": round(fro, 4),
        **ml,
        "tvd_per_column": {k: round(v, 4) for k, v in tvd_per_col.items()},
    }


def aggregate(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "tvd_per_column"} for r in results])
    metric_cols = [c for c in df.columns if c not in ("epsilon", "seed")]
    return df.groupby("epsilon")[metric_cols].agg(["mean", "std"]).reset_index()


def plot_utility_curves(agg: pd.DataFrame, out_path: Path) -> None:
    eps = agg["epsilon"].values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.errorbar(eps, agg[("avg_TVD", "mean")].values, yerr=agg[("avg_TVD", "std")].values,
                marker="o", color="#d62728", capsize=3)
    ax.set_xscale("log"); ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps])
    ax.set_xlabel("eps"); ax.set_ylabel("avg TVD (lower = better)")
    ax.set_title("Marginal Fidelity"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(eps, agg[("corr_frobenius_diff", "mean")].values, yerr=agg[("corr_frobenius_diff", "std")].values,
                marker="s", color="#9467bd", capsize=3)
    ax.set_xscale("log"); ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps])
    ax.set_xlabel("eps"); ax.set_ylabel("Frobenius diff (lower = better)")
    ax.set_title("Correlation Fidelity"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for m, color in [("LR", "#2ca02c"), ("RF", "#ff7f0e")]:
        ax.errorbar(eps, agg[(f"TSTR_{m}_acc", "mean")].values, yerr=agg[(f"TSTR_{m}_acc", "std")].values,
                    marker="o", color=color, capsize=3, label=f"TSTR {m}")
        trtr = float(agg[(f"TRTR_{m}_acc", "mean")].iloc[0])
        ax.axhline(trtr, linestyle="--", color=color, alpha=0.5, label=f"TRTR {m} = {trtr:.3f}")
    ax.set_xscale("log"); ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps])
    ax.set_xlabel("eps"); ax.set_ylabel("Accuracy")
    ax.set_title("Downstream Accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for m, color in [("LR", "#2ca02c"), ("RF", "#ff7f0e")]:
        ax.errorbar(eps, agg[(f"TSTR_{m}_f1", "mean")].values, yerr=agg[(f"TSTR_{m}_f1", "std")].values,
                    marker="^", color=color, capsize=3, label=f"TSTR {m}")
        trtr = float(agg[(f"TRTR_{m}_f1", "mean")].iloc[0])
        ax.axhline(trtr, linestyle="--", color=color, alpha=0.5, label=f"TRTR {m} = {trtr:.3f}")
    ax.set_xscale("log"); ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps])
    ax.set_xlabel("eps"); ax.set_ylabel("F1")
    ax.set_title("Downstream F1"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("PrivBayes Utility vs eps (mean +/- std over 3 seeds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    orig_df = pd.read_csv(ORIG_PATH)
    print(f"input: {orig_df.shape}, eps={EPSILONS}, seeds={SEEDS}")

    all_results = []
    for eps in EPSILONS:
        for sd in SEEDS:
            print(f"  eval eps={eps}, seed={sd}...", end=" ", flush=True)
            r = evaluate_one(orig_df, eps, sd)
            all_results.append(r)
            print(f"TVD={r['avg_TVD']:.3f}, RF={r['TSTR_RF_acc']:.3f}, LR={r['TSTR_LR_acc']:.3f}")

    print("\nplotting marginals/correlation (seed=42 representative)...")
    for eps in EPSILONS:
        synth_df = pd.read_csv(SYNTH_DIR / f"synthetic_eps{eps}_seed{SEEDS[0]}.csv")[orig_df.columns.tolist()]
        plot_marginals(orig_df, synth_df, eps, FIG_DIR / f"marginal_eps{eps}")
        plot_correlation(orig_df, synth_df, eps, FIG_DIR)

    (METRICS_DIR / "utility_detail.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    agg = aggregate(all_results)
    agg_flat = pd.DataFrame({"epsilon": agg["epsilon"].values.ravel()})
    for col in [c for c in agg.columns if c != ("epsilon", "")]:
        if isinstance(col, tuple):
            metric, stat = col
            agg_flat[f"{metric}_{stat}"] = agg[col].values
    agg_flat.to_csv(METRICS_DIR / "utility_summary.csv", index=False)

    plot_utility_curves(agg, FIG_DIR / "utility_vs_epsilon.png")

    print("\n--- summary ---")
    for s in ["avg_TVD", "corr_frobenius_diff", "TSTR_LR_acc", "TSTR_RF_acc", "TSTR_LR_f1", "TSTR_RF_f1"]:
        m = agg[(s, "mean")].values
        sd = agg[(s, "std")].values
        line = f"  {s:>22}: " + " | ".join(f"eps={e}: {mv:.4f}+/-{sv:.4f}" for e, mv, sv in zip(EPSILONS, m, sd))
        print(line)
    for m in ["LR", "RF"]:
        print(f"  TRTR {m}: acc={agg[(f'TRTR_{m}_acc', 'mean')].iloc[0]:.4f}, "
              f"f1={agg[(f'TRTR_{m}_f1', 'mean')].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
