"""Build every report figure from saved result files.  Run: python -m scripts.make_figures
Outputs report/figures/fig{1..5}_*.pdf and .png. Figures whose inputs are missing are skipped."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, OUT = Path("results"), Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "figure.dpi": 150})
J = lambda p: json.load(open(p))
JL = lambda p: [json.loads(l) for l in open(p)]
STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def fig1_dpo():
    log = JL(R / "task1_dpo/standard_train_log.jsonl")[:-1]              # drop the final partial window
    forks = [r for r in J(R / "task1_dpo/beta_sweep.json") if r["run"].startswith("beta_")]
    ls = {r["model"]: r for r in J(R / "task1_dpo/length_study.json")["models"]}
    disp = J(R / "task1_dpo/likelihood_displacement.json")
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.5))
    s = np.array([r["step"] for r in log])[4:]
    ax[0].plot(s, smooth([r["loss"] for r in log]), label="loss")
    ax[0].axhline(np.log(2), ls="--", c="gray", lw=0.8)
    t = ax[0].twinx()
    t.plot(s, smooth([r["pref_acc"] for r in log]), c="C1")
    ax[0].set(title="(a) Standard DPO training (5-step mean)", xlabel="optimizer step", ylabel="DPO loss")
    t.set_ylabel("train preference acc.", color="C1")
    x = np.arange(len(forks))
    ax[1].bar(x - 0.2, [r["heldout_pref_accuracy"] for r in forks], 0.4, label="held-out acc.")
    t = ax[1].twinx()
    t.bar(x + 0.2, [r["heldout_margin_mean"] for r in forks], 0.4, color="C2")
    ax[1].set_xticks(x, [f"β={r['beta']:g}" for r in forks])
    ax[1].set(ylim=(0.5, 0.7), title="(b) β forks (600 pairs each)", ylabel="held-out pref. acc.")
    t.set_ylabel("mean margin $m_θ$", color="C2")
    xs = np.arange(3)
    for k, m in enumerate(["standard", "length_balanced"]):
        ax[2].bar(xs + (k - 0.5) * 0.4, [ls[m][f"acc_{st}"] for st in STRATA], 0.4, label=m.replace("_", "-"))
    ax[2].axhline(0.5, ls="--", c="gray", lw=0.8)
    ax[2].set_xticks(xs, ["pref.\nlonger", "matched", "rej.\nlonger"])
    ax[2].set(title="(c) Held-out acc. by length stratum", ylim=(0.3, 0.9))
    ax[2].legend()
    w = 0.2
    for k, (name, lab) in enumerate([("standard_strat", "std"), ("length_balanced_strat", "bal")]):
        ax[3].bar(xs + (2 * k - 1.5) * w, [disp[name][st]["chosen_logratio_mean"] for st in STRATA], w, color=f"C{k}", label=f"{lab} chosen")
        ax[3].bar(xs + (2 * k - 0.5) * w, [disp[name][st]["rejected_logratio_mean"] for st in STRATA], w, color=f"C{k}", alpha=0.45, hatch="//", label=f"{lab} rejected")
    ax[3].axhline(0, c="k", lw=0.6)
    ax[3].set_xticks(xs, ["pref.\nlonger", "matched", "rej.\nlonger"])
    ax[3].set(title="(d) Likelihood displacement", ylabel="mean log π/π_ref")
    ax[3].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    save(fig, "fig1_dpo")


def fig2_ppo():
    log = JL(R / "task2_ppo/standard_train_log.jsonl")
    cached = J(R / "task2_ppo/clipping_cached_batch.json")
    kl = J(R / "task2_ppo/kl_forks.json")["rows"]
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.5))
    u = [r["update"] for r in log]
    ax[0].plot(u, [r["reward_effective"] for r in log], "o-", ms=2.5, label="reward (effective)")
    ax[0].plot(u, [r["kl_token_mean"] * 1000 for r in log], "s-", ms=2.5, label="KL/token ×10³")
    ax[0].set(title="(a) Standard PPO continuation", xlabel="update")
    ax[0].legend(loc="lower left")
    ax[1].plot(u, [r["value_loss"] for r in log], "o-", ms=2.5, label="value loss")
    t = ax[1].twinx()
    t.plot(u, [r["value_explained_variance"] for r in log], "s-", ms=2.5, c="C3")
    t.set_yscale("symlog")
    t.set_ylabel("critic explained var. (symlog)", color="C3")
    ax[1].set(title="(b) Critic behaviour", xlabel="update", ylabel="value loss")
    pe = cached["per_epsilon"]
    x = np.arange(len(pe))
    ax[2].bar(x - 0.2, [p["clip_fraction_outside_band"] for p in pe], 0.4, label="outside [1−ε,1+ε]")
    ax[2].bar(x + 0.2, [p["affected_token_fraction_binding"] for p in pe], 0.4, label="clipping binds")
    ax[2].set_xticks(x, [f"ε={p['epsilon']:g}" for p in pe])
    ax[2].set(title="(c) Cached batch (8,814 tokens)", ylabel="token fraction")
    ax[2].legend()
    x = np.arange(len(kl))
    ax[3].bar(x, [r["heldout_reward_effective"] for r in kl],
              yerr=[r["heldout_reward_effective_std"] / np.sqrt(32) for r in kl], capsize=3)
    ax[3].set_xticks(x, [f"β_KL={r['kl_beta']:g}" for r in kl])
    ax[3].set(title="(d) KL forks: held-out reward", ylabel="effective reward", ylim=(1.0, 2.4))
    save(fig, "fig2_ppo")


def fig3_grpo():
    log = JL(R / "task3_grpo/standard_train_log.jsonl")
    gs = J(R / "task3_grpo/group_size_study.json")["results"]
    nc = J(R / "task3_grpo/normalization_comparison.json")["runs"]
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.5))
    u = [r["update"] for r in log]
    ax[0].plot(u, [r["reward_mean"] for r in log], "o-", ms=2.5, label="group mean reward")
    ax[0].plot(u, [r["group_reward_std_mean"] for r in log], "s-", ms=2.5, label="within-group std")
    for r in log:
        if r["uninformative_group_frac"] > 0:
            ax[0].axvline(r["update"], c="red", alpha=0.3, lw=3)
    ax[0].set(title="(a) Standard GRPO (red = uninformative)", xlabel="update")
    ax[0].legend(loc="lower left")
    Ks = sorted({r["K"] for r in gs})
    bins = ["all", "hard", "medium", "easy"]
    x = np.arange(len(Ks))
    for k, b in enumerate(bins):
        ax[1].bar(x + (k - 1.5) * 0.2, [next(r for r in gs if r["K"] == K and r["difficulty"] == b)["low_signal_rate_std_lt_0p1"] for K in Ks], 0.2, label=b)
    ax[1].set_xticks(x, [f"K={K}" for K in Ks])
    ax[1].set(title="(b) Low-signal groups (std<0.1)", ylabel="fraction of groups")
    ax[1].legend()
    for k, b in enumerate(bins):
        ax[2].plot(Ks, [next(r for r in gs if r["K"] == K and r["difficulty"] == b)["sign_agreement_with_k8"] for K in Ks], "o-", ms=3, label=b)
    ax[2].set(title="(c) Advantage sign agreement with K=8", xlabel="group size K", ylabel="agreement", xticks=Ks)
    ax[2].legend()
    lb = ["short", "medium", "long"]
    x = np.arange(3)
    for k, (run, lab) in enumerate([("norm_grpo", "canonical 1/T"), ("norm_dr_grpo", "Dr. GRPO 1/L")]):
        ax[3].bar(x + (k - 0.5) * 0.4, [nc[run]["allocation_by_length"][b]["share_of_total_weight"] for b in lb], 0.4, label=lab)
    ax[3].set_xticks(x, lb)
    ax[3].set(title="(d) Share of gradient weight by length", ylabel="share")
    ax[3].legend()
    save(fig, "fig3_grpo")


def fig4_safety():
    rows = list(csv.DictReader(open(R / "task4_safety/category_label_counts.csv")))
    audit = J(R / "task4_safety/safety_summary.json").get("manual_audit")
    labels = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
    pols = ["sft", "dpo", "ppo", "grpo"]
    fig, ax = plt.subplots(1, 2 if not audit else 3, figsize=(11, 2.6))
    for a, cls in zip(ax[:2], ["SAFE", "UNSAFE"]):
        bottom = np.zeros(len(pols))
        for lab in labels:
            v = np.array([sum(int(r[lab]) for r in rows if r["policy"] == p and r["benchmark_class"] == cls) for p in pols], float)
            tot = np.array([sum(int(r["n"]) for r in rows if r["policy"] == p and r["benchmark_class"] == cls) for p in pols], float)
            a.bar(pols, v / tot, bottom=bottom, label=lab.replace("_", " ").lower())
            bottom += v / tot
        a.set(title=f"({'a' if cls == 'SAFE' else 'b'}) Judge labels, {cls} prompts", ylabel="fraction")
    ax[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if audit:
        cm = np.array(audit["confusion_rows_manual_cols_judge"]["matrix"])
        ax[2].imshow(cm, cmap="Blues")
        short = ["SAFE", "J.REF", "U.COMP", "O.REF", "AMB"]
        ax[2].set_xticks(range(5), short, rotation=45)
        ax[2].set_yticks(range(5), short)
        for i in range(5):
            for j in range(5):
                ax[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        ax[2].set(title=f"(c) Manual (rows) vs judge (cols), κ={audit['cohen_kappa']:.2f}")
    save(fig, "fig4_safety")


def fig5_feedback():
    gsm, tr = J(R / "task5_feedback/eval_gsm_summary.json"), J(R / "task5_feedback/eval_transfer_summary.json")
    diag = J(R / "task5_feedback/diagnostics_summary.json")["by_category"]
    pols = ["sft", "rlvr", "rlaif"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
    x = np.arange(3)
    ax[0].bar(x - 0.2, [gsm["policies"][p]["exact_accuracy"] for p in pols], 0.4, label="GSM8K (in-domain)")
    ax[0].bar(x + 0.2, [tr["policies"][p]["exact_accuracy"] for p in pols], 0.4, label="SVAMP (transfer)")
    ax[0].set_xticks(x, [p.upper() for p in pols])
    ax[0].set(title="(a) Exact-answer accuracy", ylabel="accuracy", ylim=(0, 0.6))
    ax[0].legend(loc="upper left")
    cats = list(diag)
    short = {"reasoning": "reason.", "outcome": "outcome", "filler": "filler", "distractor": "distract."}
    lab = [f"{short[c]}\n{'verif.' if m == 'verifier' else 'judge'}" for c in cats for m in ["verifier", "judge"]]
    vals = [diag[c][m] for c in cats for m in ["verifier", "judge"]]
    b = np.zeros(len(vals))
    for key, col, name in [("better_rate", "C2", "better"), ("tie_rate", "C7", "tie"), ("wrong_preference_rate", "C3", "wrong")]:
        v = np.array([d[key] for d in vals])
        ax[1].bar(range(len(vals)), v, bottom=b, color=col, label=name)
        b += v
    ax[1].set_xticks(range(len(vals)), lab, fontsize=6)
    ax[1].set(title="(b) Controlled diagnostics: clean vs perturbed", ylabel="fraction of pairs")
    ax[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "fig5_feedback")


if __name__ == "__main__":
    for f in [fig1_dpo, fig2_ppo, fig3_grpo, fig4_safety, fig5_feedback]:
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: missing {e.filename}")
