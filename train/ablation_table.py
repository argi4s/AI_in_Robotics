import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
EVAL_WINDOW = 200


def load_npy(path):
    """Load a .npy file; return None if missing."""
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def sr_from_successes(arr, window=EVAL_WINDOW):
    if arr is None:
        return float("nan")
    return float(np.mean(arr[-window:]))


def avg_steps_from_steps(arr, window=EVAL_WINDOW):
    if arr is None:
        return float("nan")
    recent = arr[-window:]
    valid  = recent[~np.isnan(recent.astype(float))]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


# ─── Load all results ─────────────────────────────────────────────────────────

# Tabular results — load from pkl if available, else use known values from
# the agent_comparison.py run (1500 eps, SHARED_SEED=42, EVAL_WINDOW=200).
TABULAR_FALLBACK = {
    "sarsa_12": (0.300, 74.2),
    "ql_12":    (0.275, 71.8),
    "sarsa_16": (0.135, float("nan")),
    "ql_16":    (0.105, float("nan")),
}
tabular = {}
pkl_path = "scalability_summary.pkl"
if not os.path.exists(pkl_path):
    pkl_path = os.path.join(RESULTS_DIR, "scalability_summary.pkl")
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    sarsa = data["sarsa_summary"]
    ql    = data["ql_summary"]
    tabular["sarsa_12"] = (sarsa[12]["success_rate"], sarsa[12]["avg_steps"])
    tabular["ql_12"]    = (ql[12]["success_rate"],    ql[12]["avg_steps"])
    tabular["sarsa_16"] = (sarsa[16]["success_rate"], sarsa[16]["avg_steps"])
    tabular["ql_16"]    = (ql[16]["success_rate"],    ql[16]["avg_steps"])
    print(f"  Loaded tabular results from {pkl_path}")
else:
    print("  scalability_summary.pkl not found — using known fallback values.")
    tabular = dict(TABULAR_FALLBACK)

# Navigation DQN (dueling)
dqn_s  = load_npy(os.path.join(RESULTS_DIR, "dqn_successes_16x16.npy"))
dqn_st = load_npy(os.path.join(RESULTS_DIR, "dqn_steps_16x16.npy"))
dqn_sr  = sr_from_successes(dqn_s)
dqn_avg = avg_steps_from_steps(dqn_st)

# Vanilla DQN ablation (config A)
vdqn_s  = load_npy(os.path.join(RESULTS_DIR, "ablation_vanilla_dqn_successes.npy"))
vdqn_st = load_npy(os.path.join(RESULTS_DIR, "ablation_vanilla_dqn_steps.npy"))
vdqn_sr  = sr_from_successes(vdqn_s)
vdqn_avg = avg_steps_from_steps(vdqn_st)

# Eval results (from eval_entity.py)
rand_s   = load_npy(os.path.join(RESULTS_DIR, "eval_random_baseline.npy"))
ora_s    = load_npy(os.path.join(RESULTS_DIR, "eval_oracle_oracle.npy"))
perc_s   = load_npy(os.path.join(RESULTS_DIR, "eval_oracle_perception.npy"))
gen_data = load_npy(os.path.join(RESULTS_DIR, "eval_generalisation.npy"))

rand_sr  = sr_from_successes(rand_s, window=len(rand_s)) if rand_s is not None else float("nan")
ora_sr   = sr_from_successes(ora_s,  window=len(ora_s))  if ora_s  is not None else float("nan")
perc_sr  = sr_from_successes(perc_s, window=len(perc_s)) if perc_s is not None else float("nan")
gen_mean = float(np.mean(gen_data)) if gen_data is not None else float("nan")
gen_std  = float(np.std(gen_data))  if gen_data is not None else float("nan")

# Entity DQN oracle (from training run)
ent_s  = load_npy(os.path.join(RESULTS_DIR, "entity_successes.npy"))
ent_st = load_npy(os.path.join(RESULTS_DIR, "entity_steps.npy"))
ent_sr  = sr_from_successes(ent_s)
ent_avg = avg_steps_from_steps(ent_st)

# Ablation B: no energy
ne_s  = load_npy(os.path.join(RESULTS_DIR, "ablation_no_energy_successes.npy"))
ne_st = load_npy(os.path.join(RESULTS_DIR, "ablation_no_energy_steps.npy"))
ne_sr  = sr_from_successes(ne_s)
ne_avg = avg_steps_from_steps(ne_st)

# Ablation C: no +5 bonus
nb_s  = load_npy(os.path.join(RESULTS_DIR, "ablation_no_bonus_successes.npy"))
nb_st = load_npy(os.path.join(RESULTS_DIR, "ablation_no_bonus_steps.npy"))
nb_sr  = sr_from_successes(nb_s)
nb_avg = avg_steps_from_steps(nb_st)


# ─── Print table ──────────────────────────────────────────────────────────────

def fmt_sr(v):
    return f"{v:.1%}" if not np.isnan(v) else "—"

def fmt_st(v):
    return f"{v:.1f}" if not np.isnan(v) else "—"

rows = [
    # (label,                           sr,                         avg_steps,               note)
    ("Random baseline (entity 12×12)",  rand_sr,                    float("nan"),            ""),
    ("SARSA 12×12",                     tabular["sarsa_12"][0],     tabular["sarsa_12"][1],  "tabular"),
    ("Q-Learning 12×12",                tabular["ql_12"][0],        tabular["ql_12"][1],     "tabular"),
    ("Vanilla DQN (no dueling) 12×12",  vdqn_sr,                   vdqn_avg,                "ablation A"),
    ("Dueling DQN 16×16 (nav only)",    dqn_sr,                    dqn_avg,                 "nav, larger maze"),
    ("Entity DQN oracle 12×12",         ent_sr,                    ent_avg,                 "full model"),
    ("Entity DQN perception 12×12",     perc_sr,                   float("nan"),            "cross-test"),
    ("Entity DQN no energy 12×12",      ne_sr,                     ne_avg,                  "ablation B"),
    ("Entity DQN no +5 bonus 12×12",    nb_sr,                     nb_avg,                  "ablation C"),
    (f"Generalisation (mean±{gen_std:.1%})", gen_mean,             float("nan"),            "unseen seeds"),
]

print("\n" + "=" * 72)
print("  ABLATION TABLE  —  Task 3 RL Evaluation")
print("=" * 72)
print(f"  {'Configuration':<38} {'SR':>8}  {'Avg Steps':>10}  {'Note':>12}")
print(f"  {'-'*68}")
for label, sr, avg, note in rows:
    print(f"  {label:<38} {fmt_sr(sr):>8}  {fmt_st(avg):>10}  {note:>12}")
print("=" * 72)

missing = [label for label, sr, _, _ in rows if np.isnan(sr)]
if missing:
    print(f"\n  Missing results (run the corresponding script first):")
    for m in missing:
        print(f"    - {m}")


# ─── Bar chart ────────────────────────────────────────────────────────────────

plot_rows = [(l, sr) for l, sr, _, _ in rows if not np.isnan(sr)]
p_labels  = [r[0] for r in plot_rows]
p_values  = [r[1] for r in plot_rows]

colour_map = {
    "Random baseline": "#888888",
    "SARSA":           "#F44336",
    "Q-Learning":      "#FF9800",
    "Vanilla DQN":     "#9C27B0",
    "Dueling DQN":     "#2196F3",
    "Entity DQN oracle": "#4CAF50",
    "Entity DQN perception": "#FF5722",
    "Entity DQN no energy":  "#795548",
    "Entity DQN no +5":      "#607D8B",
    "Generalisation":  "#009688",
}

def pick_colour(label):
    for k, c in colour_map.items():
        if k.lower() in label.lower():
            return c
    return "#90A4AE"

colours = [pick_colour(l) for l in p_labels]

short_labels = [l.replace(" 12×12", "").replace("(", "\n(") for l in p_labels]

fig, ax = plt.subplots(figsize=(13, 5))
xs   = np.arange(len(p_labels))
bars = ax.bar(xs, p_values, color=colours, edgecolor="white", width=0.65, alpha=0.88)

for bar, v in zip(bars, p_values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(short_labels, fontsize=8.5, rotation=15, ha="right")
ax.set_ylabel("Success Rate (last 200 episodes)", fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_title("Ablation Study — Task 3 RL\nEach bar isolates one design decision",
             fontsize=12, fontweight="bold")
ax.axhline(rand_sr if not np.isnan(rand_sr) else 0,
           color="#888888", linestyle="--", linewidth=1, alpha=0.6,
           label="Random baseline")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ablation_table.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {RESULTS_DIR}/ablation_table.png")
print("Done.")
