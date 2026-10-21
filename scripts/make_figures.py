#!/usr/bin/env python3
"""Phase 3: thesis figures from the Script-4 CSVs.

Reads the CSVs written by scripts/evaluate.py (results/) and renders PNGs to
results/figures/. Every figure is guarded: a missing or empty CSV is skipped with
a notice rather than crashing, so the same command works on a mini-run (some
secondary CSVs empty) and the full run.

Continuation-mode experiment: k in {1,2,3}; error_types
{off_topic, paraphrased_shift, false_consensus}; models 7B/14B/32B.

Primary headline figures use results_per_condition.csv (Susceptibility, Recovery
Gain). Secondary/exploratory figures use the derived CSVs (token_economy,
decoy_anchoring, length_bins, confidence, cost_throughput). Per-sample throughput
is labelled approx_ (vLLM batches in parallel; not a true per-sample latency).

Usage:
  python scripts/make_figures.py
  python scripts/make_figures.py --results-dir results --figures-dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display needed on the cluster
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ERROR_TYPES = ["off_topic", "paraphrased_shift", "false_consensus"]


def _load(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.exists():
        print(f"[SKIP] {name} not found")
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        print(f"[SKIP] {name} empty/unreadable")
        return None
    if df.empty:
        print(f"[SKIP] {name} has no rows")
        return None
    return df


def _save(fig, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def _short_model(name: object) -> str:
    s = str(name)
    for tag in ("32B", "14B", "7B", "32b", "14b", "7b"):
        if tag in s:
            return tag.upper()
    return s.split("/")[-1][:16]


# ---------------------------------------------------------------------------
# Primary: Susceptibility / Recovery Gain vs k (per model x error_type)
# ---------------------------------------------------------------------------
def fig_main_effects(results_dir: Path, figures_dir: Path) -> None:
    df = _load(results_dir, "results_per_condition.csv")
    if df is None:
        return
    df = df[df.get("domain").isin(["math"]) | df.get("domain").isna()] if "domain" in df else df
    for metric, title in (
        ("susceptibility", "Susceptibility (1 - acc_stop)"),
        ("recovery_gain", "Recovery Gain (acc_recovery - acc_stop)"),
    ):
        if metric not in df.columns:
            continue
        models = sorted(df["model"].dropna().unique(), key=_short_model)
        n = len(models)
        fig, axes = plt.subplots(1, max(n, 1), figsize=(4.2 * max(n, 1), 4), squeeze=False)
        for ax, model in zip(axes[0], models):
            sub = df[df["model"] == model]
            for et in ERROR_TYPES:
                ets = sub[sub["error_type"] == et].sort_values("k")
                if ets.empty:
                    continue
                ax.plot(ets["k"], ets[metric], marker="o", label=et)
            ax.set_title(_short_model(model))
            ax.set_xlabel("k (injections)")
            ax.set_ylabel(metric)
            ax.set_xticks(sorted(sub["k"].dropna().unique()))
            ax.grid(True, alpha=0.3)
        axes[0][0].legend(fontsize=8, title="error_type")
        fig.suptitle(title)
        _save(fig, figures_dir, f"main_{metric}_vs_k.png")


# ---------------------------------------------------------------------------
# Secondary: Recovery-inflation distribution (token ratio) per model
# ---------------------------------------------------------------------------
def fig_recovery_inflation(results_dir: Path, figures_dir: Path) -> None:
    df = _load(results_dir, "token_economy_per_dialogue.csv")
    if df is None or "recovery_inflation_ratio_tokens" not in df.columns:
        return
    df = df.dropna(subset=["recovery_inflation_ratio_tokens"])
    if df.empty:
        print("[SKIP] no recovery_inflation_ratio_tokens values")
        return
    models = sorted(df["model"].dropna().unique(), key=_short_model)
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4))
    for model in models:
        vals = df[df["model"] == model]["recovery_inflation_ratio_tokens"].astype(float)
        vals = vals[vals > 0]
        if vals.empty:
            continue
        ax_hist.hist(vals, bins=30, alpha=0.5, label=_short_model(model))
        sv = vals.sort_values().to_numpy()
        cdf = (sv.argsort().argsort() + 1) / len(sv)
        order = sv.argsort()
        ax_cdf.plot(sv[order], cdf[order], label=_short_model(model))
        ax_cdf.axvline(float(vals.quantile(0.95)), ls="--", alpha=0.4)
    ax_hist.axvline(1.0, color="k", ls=":", label="ratio=1 (baseline)")
    ax_hist.set_xlabel("recovery_inflation_ratio_tokens")
    ax_hist.set_ylabel("dialogues")
    ax_hist.set_title("Recovery inflation (tokens)")
    ax_hist.legend(fontsize=8)
    ax_cdf.set_xlabel("recovery_inflation_ratio_tokens")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("CDF (dashed = per-model p95)")
    ax_cdf.grid(True, alpha=0.3)
    _save(fig, figures_dir, "recovery_inflation.png")


# ---------------------------------------------------------------------------
# Secondary: Decoy anchoring (false_consensus) — stop vs recovery adoption
# ---------------------------------------------------------------------------
def fig_decoy_anchoring(results_dir: Path, figures_dir: Path) -> None:
    df = _load(results_dir, "decoy_anchoring.csv")
    if df is None:
        return
    needed = {"stop_decoy_adoption_rate", "recovery_decoy_adoption_rate"}
    if not needed.issubset(df.columns):
        print("[SKIP] decoy_anchoring missing adoption columns")
        return
    df = df.copy()
    df["cell"] = (
        df.apply(lambda r: f"{_short_model(r['model'])} k{r['k']} {r['length_level']}", axis=1)
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(df))
    width = 0.4
    ax1.bar([i - width / 2 for i in x], df["stop_decoy_adoption_rate"], width, label="stop")
    ax1.bar([i + width / 2 for i in x], df["recovery_decoy_adoption_rate"], width, label="recovery")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["cell"], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("decoy adoption rate")
    ax1.set_title("Decoy adoption: stop vs recovery (false_consensus)")
    ax1.legend(fontsize=8)
    if {"decoy_escape_rate", "decoy_regression_rate"}.issubset(df.columns):
        ax2.bar([i - width / 2 for i in x], df["decoy_escape_rate"].fillna(0), width, label="escape")
        ax2.bar([i + width / 2 for i in x], df["decoy_regression_rate"].fillna(0), width, label="regression")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(df["cell"], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("rate")
        ax2.set_title("Escape (stop=decoy) vs regression (stop!=decoy)")
        ax2.legend(fontsize=8)
    _save(fig, figures_dir, "decoy_anchoring.png")


# ---------------------------------------------------------------------------
# Secondary: Length bins (help/harm/accuracy by recovery-length quartile)
# ---------------------------------------------------------------------------
def fig_length_bins(results_dir: Path, figures_dir: Path) -> None:
    df = _load(results_dir, "length_bins.csv")
    if df is None:
        return
    metrics = sorted(df["metric_name"].dropna().unique())
    bin_types = sorted(df["bin_type"].dropna().unique())
    if not metrics or not bin_types:
        return
    fig, axes = plt.subplots(
        len(bin_types), 1, figsize=(8, 3.6 * len(bin_types)), squeeze=False
    )
    for ax, bt in zip(axes[:, 0], bin_types):
        sub = df[df["bin_type"] == bt]
        for model in sorted(sub["model"].dropna().unique(), key=_short_model):
            for metric in metrics:
                ms = sub[(sub["model"] == model) & (sub["metric_name"] == metric)].sort_values("bin_index")
                if ms.empty:
                    continue
                ax.plot(ms["bin_index"], ms["value"], marker="o",
                        label=f"{_short_model(model)} · {metric.replace('_in_bin','')}")
        ax.set_title(f"{bt}-length quartiles (associative)")
        ax.set_xlabel(f"{bt} quartile (1=short .. 4=long)")
        ax.set_ylabel("rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    _save(fig, figures_dir, "length_bins.png")


# ---------------------------------------------------------------------------
# Exploratory: confidence (mean_logprob stop vs recovery)
# ---------------------------------------------------------------------------
def fig_confidence(results_dir: Path, figures_dir: Path) -> None:
    df = _load(results_dir, "confidence.csv")
    if df is None or not {"stop_mean_logprob", "recovery_mean_logprob"}.issubset(df.columns):
        return
    df = df.dropna(subset=["stop_mean_logprob", "recovery_mean_logprob"], how="all")
    if df.empty:
        print("[SKIP] confidence has no logprob values")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df = df.copy()
    df["cell"] = df.apply(
        lambda r: f"{_short_model(r['model'])} {r.get('error_type')} k{r.get('k')}", axis=1
    )
    x = range(len(df))
    width = 0.4
    ax.bar([i - width / 2 for i in x], df["stop_mean_logprob"], width, label="stop")
    ax.bar([i + width / 2 for i in x], df["recovery_mean_logprob"], width, label="recovery")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["cell"], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("mean log-prob (higher = more confident)")
    ax.set_title("Confidence: stop vs recovery")
    ax.legend(fontsize=8)
    _save(fig, figures_dir, "confidence.png")


# ---------------------------------------------------------------------------
# Operational: cost / throughput per model
# ---------------------------------------------------------------------------
def fig_cost_throughput(results_dir: Path, figures_dir: Path) -> None:
    df = _load(results_dir, "cost_throughput.csv")
    if df is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    labels = [_short_model(m) for m in df["model"]]
    if "tokens_per_second_batch_mean" in df.columns:
        ax1.bar(labels, df["tokens_per_second_batch_mean"])
        ax1.set_title("Throughput (tokens/s, batch)")
        ax1.set_ylabel("tokens/s")
    if "dialogue_wall_time_s_mean" in df.columns:
        ax2.bar(labels, df["dialogue_wall_time_s_mean"])
        ax2.set_title("Mean dialogue wall time")
        ax2.set_ylabel("seconds")
    _save(fig, figures_dir, "cost_throughput.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render thesis figures from Script-4 CSVs.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir
    figures_dir = args.figures_dir or (results_dir / "figures")

    print(f"[INFO] results_dir={results_dir} figures_dir={figures_dir}")
    fig_main_effects(results_dir, figures_dir)
    fig_recovery_inflation(results_dir, figures_dir)
    fig_decoy_anchoring(results_dir, figures_dir)
    fig_length_bins(results_dir, figures_dir)
    fig_confidence(results_dir, figures_dir)
    fig_cost_throughput(results_dir, figures_dir)
    print("[DONE] figures written to", figures_dir)


if __name__ == "__main__":
    main()
