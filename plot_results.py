#plot_results.py
"""
Plot test-set results for protein stability regression, report top sequences,
and plot stability score distributions for train/valid/test.

Inputs
  - preds.npy : model predictions (float32, shape [N])
  - y_true.npy: ground truth values for TEST (float32, shape [N])
  - test_seqs.txt: one AA sequence per line (no headers) [optional, for top-20 table]
  - y_train.npy, y_valid.npy: optional paths for KDE plot

Outputs (in --outdir)
  - fig_parity_test.png
  - fig_abs_error_bins.png
  - fig_stability_kde.png      
  - metrics.json
  - top_sequences.tsv         
"""

import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Seaborn for KDE
import seaborn as sns

# ------------- utils -------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(y_true, y_pred).correlation
        return float(0.0 if rho is None else rho)
    except Exception:
        y_rank = np.argsort(np.argsort(y_true))
        p_rank = np.argsort(np.argsort(y_pred))
        y = y_rank - y_rank.mean(); p = p_rank - p_rank.mean()
        denom = np.sqrt((y**2).sum()) * np.sqrt((p**2).sum())
        return float((y*p).sum() / denom) if denom > 0 else 0.0

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    rho = _spearman_safe(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "spearman": rho, "n": int(len(y_true))}

# ------------- plots -------------
def parity_plot(y_true, y_pred, out_path, title_prefix="Predicted vs true stability",
                use_hexbin_threshold=50):
    m = _metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.20, top=0.88)

    xmin = float(min(y_true.min(), y_pred.min()))
    xmax = float(max(y_true.max(), y_pred.max()))
    pad = 0.02 * (xmax - xmin + 1e-9)
    lims = [xmin - pad, xmax + pad]

    n = len(y_true)
    if n >= use_hexbin_threshold:
        hb = ax.hexbin(y_true, y_pred, gridsize=50, mincnt=1, bins="log", cmap="viridis")
        cbax = inset_axes(ax, width="4%", height="60%", loc="upper right", borderpad=0.8)
        cbar = fig.colorbar(hb, cax=cbax)
        cbar.set_label("log10 count")
    else:
        ax.scatter(y_true, y_pred, s=9, alpha=0.35)

    ax.plot(lims, lims, lw=1, color="C0")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True stability")
    ax.set_ylabel("Predicted stability")
    ax.set_title(title_prefix, fontsize=10.5, pad=10)

    txt = f"ρ={m['spearman']:.3f}\nMAE={m['mae']:.3f}\nRMSE={m['rmse']:.3f}\nn={m['n']}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
            fontsize=9)

    fig.savefig(out_path)
    plt.close(fig)


def abs_error_bins(y_true, y_pred, out_path,
                   bins=(0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, np.inf),
                   labels=("≤0.1","0.1–0.2","0.2–0.3","0.3–0.5","0.5–0.75","0.75–1.0",">1.0"),
                   title="Absolute error histogram with cumulative coverage"):
    err = np.abs(y_pred - y_true)
    hist, _ = np.histogram(err, bins=bins)

    freq = hist.astype(np.float64)
    if freq.sum() == 0:
        freq = np.zeros_like(freq); cum = np.zeros_like(freq)
    else:
        freq = freq / freq.sum(); cum = np.cumsum(freq)

    x = np.arange(len(freq))

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.20, top=0.88)

    ax.bar(x, freq, width=0.65, color="#1f77b4")
    ax.set_ylabel("Fraction of samples")
    ax.set_xlabel("|error| bin")
    ax.set_ylim(0, max(0.3, float(freq.max()) + 0.05))
    ax.margins(x=0.02)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)

    ax2 = ax.twinx()
    ax2.plot(x, cum, marker="o", color="#d62728", linewidth=2, markersize=4)
    ax2.set_ylabel("Cumulative coverage")
    ax2.tick_params(axis="y")
    ax2.set_ylim(0, 1.02)

    ax.set_title(title, fontsize=10.5, pad=10)
    fig.savefig(out_path)
    plt.close(fig)


def kde_stability(train_y, valid_y, test_y, out_path):
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=300)
        if train_y is not None and len(train_y) > 0:
            sns.kdeplot(x=train_y, fill=True, common_norm=False, label="train", ax=ax)
        if valid_y is not None and len(valid_y) > 0:
            sns.kdeplot(x=valid_y, fill=True, common_norm=False, label="valid", ax=ax)
        if test_y is not None and len(test_y) > 0:
            sns.kdeplot(x=test_y,  fill=True, common_norm=False, label="test",  ax=ax)
        ax.set_xlabel("stability score")
        ax.set_ylabel("density")
        ax.set_title("Stability score distribution by split", fontsize=10.5, pad=10)
        ax.legend(title="split")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


# ------------- main -------------
def _load_sequences(path: str):
    with open(path, "r") as f:
        seqs = [ln.strip() for ln in f if ln.strip() and not ln.startswith(">")]
    return seqs

def _load_npy_or_none(p):
    return np.load(p).astype(np.float32).ravel() if p and os.path.exists(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, default="preds.npy", help="Path to preds.npy")
    ap.add_argument("--truth", type=str, default="y_true.npy", help="Path to y_true.npy (test labels)")
    ap.add_argument("--seqs", type=str, default=None, help="Path to test_seqs.txt with one sequence per line")
    # new optional label paths for KDE
    ap.add_argument("--y_train", type=str, default=None, help="Path to y_train_aligned.npy for KDE")
    ap.add_argument("--y_valid", type=str, default=None, help="Path to y_valid_aligned.npy for KDE")
    # outdir and misc
    ap.add_argument("--outdir", type=str, default="figs", help="Output directory for figures and tables")
    ap.add_argument("--hexbin_threshold", type=int, default=50, help="Use hexbin if N >= threshold")
    args = ap.parse_args()

    if not os.path.exists(args.preds) or not os.path.exists(args.truth):
        print(f"Missing required files. Got preds={args.preds} truth={args.truth}", file=sys.stderr)
        sys.exit(1)

    y_pred = np.load(args.preds).astype(np.float32).ravel()
    y_true = np.load(args.truth).astype(np.float32).ravel()
    if len(y_pred) != len(y_true):
        print(f"Length mismatch: preds={len(y_pred)} truth={len(y_true)}", file=sys.stderr)
        sys.exit(1)

    seqs = None
    if args.seqs is not None:
        if not os.path.exists(args.seqs):
            print(f"[WARN] --seqs provided but not found: {args.seqs}. Skipping sequence ranking.", file=sys.stderr)
        else:
            seqs = _load_sequences(args.seqs)
            if len(seqs) != len(y_true):
                print(f"[WARN] Sequence count {len(seqs)} != labels {len(y_true)}. Skipping sequence ranking.", file=sys.stderr)
                seqs = None

    _ensure_dir(args.outdir)

    # save metrics summary
    m = _metrics(y_true, y_pred)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    # plots
    parity_plot(
        y_true, y_pred,
        out_path=os.path.join(args.outdir, "fig_parity_test.png"),
        title_prefix="Predicted vs true stability (test set)",
        use_hexbin_threshold=args.hexbin_threshold,
    )
    abs_error_bins(
        y_true, y_pred,
        out_path=os.path.join(args.outdir, "fig_abs_error_bins.png"),
    )

    # KDE distribution plot across splits (train, valid, test)
    y_train = _load_npy_or_none(args.y_train)
    y_valid = _load_npy_or_none(args.y_valid)
    kde_out = os.path.join(args.outdir, "fig_stability_kde.png")
    if any(v is not None for v in [y_train, y_valid, y_true]):
        kde_stability(y_train, y_valid, y_true, kde_out)

    # top 20 best and worst by absolute error
    if seqs is not None:
        abs_err = np.abs(y_pred - y_true)
        order = np.argsort(abs_err)
        best_idx = order[:20]
        worst_idx = order[-20:][::-1]

        print("\nTop 20 best predictions by |error|:")
        for rank, i in enumerate(best_idx, 1):
            print(f"{rank:2d}. idx={i:7d}  |err|={abs_err[i]:.6f}  y_true={y_true[i]:.6f}  y_pred={y_pred[i]:.6f}  seq={seqs[i]}")

        print("\nTop 20 worst predictions by |error|:")
        for rank, i in enumerate(worst_idx, 1):
            print(f"{rank:2d}. idx={i:7d}  |err|={abs_err[i]:.6f}  y_true={y_true[i]:.6f}  y_pred={y_pred[i]:.6f}  seq={seqs[i]}")

        tsv_path = os.path.join(args.outdir, "top_sequences.tsv")
        with open(tsv_path, "w") as f:
            f.write("kind\trank\tindex\tabs_error\ty_true\ty_pred\tsequence\n")
            for rank, i in enumerate(best_idx, 1):
                f.write(f"best\t{rank}\t{i}\t{abs_err[i]:.6f}\t{y_true[i]:.6f}\t{y_pred[i]:.6f}\t{seqs[i]}\n")
            for rank, i in enumerate(worst_idx, 1):
                f.write(f"worst\t{rank}\t{i}\t{abs_err[i]:.6f}\t{y_true[i]:.6f}\t{y_pred[i]:.6f}\t{seqs[i]}\n")
        print(f"\nSaved ranking table: {tsv_path}")

    print(f"\nDone. Saved figures and metrics in: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
