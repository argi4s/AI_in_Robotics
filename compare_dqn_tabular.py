import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from models.dqn_network import DuelingDQN


# ─── Plot colours ─────────────────────────────────────────────────────────────
SARSA_COLOR = "#2196F3"   # blue
QL_COLOR    = "#F44336"   # red
DQN_COLOR   = "#4CAF50"   # green

DQN_GRID_SIZE = 16
EVAL_WINDOW   = 200


# ─── Hardcoded tabular results ────────────────────────────────────────────────
# These are the results from the student's agent_comparison.py run
# (1500 episodes per grid size, SHARED_SEED=42).
# Automatically overridden if scalability_summary.pkl is found.
KNOWN_TABULAR = {
    "grid_sizes": [8, 10, 12, 14, 16],
    "n_episodes": 1500,
    "eval_window": 200,
    "sarsa_summary": {
        8:  {"success_rate": 0.675, "q_table_size": 110, "avg_steps": float("nan")},
        10: {"success_rate": 0.470, "q_table_size": 212, "avg_steps": float("nan")},
        12: {"success_rate": 0.300, "q_table_size": 343, "avg_steps": float("nan")},
        14: {"success_rate": 0.180, "q_table_size": 506, "avg_steps": float("nan")},
        16: {"success_rate": 0.135, "q_table_size": 702, "avg_steps": float("nan")},
    },
    "ql_summary": {
        8:  {"success_rate": 0.750, "q_table_size": 111, "avg_steps": float("nan")},
        10: {"success_rate": 0.505, "q_table_size": 211, "avg_steps": float("nan")},
        12: {"success_rate": 0.275, "q_table_size": 342, "avg_steps": float("nan")},
        14: {"success_rate": 0.165, "q_table_size": 504, "avg_steps": float("nan")},
        16: {"success_rate": 0.105, "q_table_size": 701, "avg_steps": float("nan")},
    },
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tabular_data() -> dict:
    """Load tabular data from pkl if available, else fall back to hardcoded values."""
    pkl = "scalability_summary.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded tabular data from {pkl}")
        return data
    print("  Using hardcoded tabular results.")
    print("  (Re-run agent_comparison.py to refresh and generate the pkl.)")
    return KNOWN_TABULAR


def load_dqn_results() -> dict:
    """
    Load DQN episode metrics saved by train/train_dqn.py.

    Returns dict with:
        successes : float array of shape (n_episodes,), values 0.0 or 1.0
        steps     : float array of shape (n_episodes,), NaN for failed episodes
    """
    s_path = os.path.join("results", "dqn_successes_16x16.npy")
    t_path = os.path.join("results", "dqn_steps_16x16.npy")
    if not os.path.exists(s_path) or not os.path.exists(t_path):
        raise FileNotFoundError(
            "DQN metrics not found.\n"
            "Run  python train/train_dqn.py  first, then retry."
        )
    return {
        "successes": np.load(s_path),
        "steps":     np.load(t_path),
    }


def compute_dqn_summary(dqn_results: dict) -> dict:
    """Compute final success rate and avg steps from DQN episode metrics."""
    successes = dqn_results["successes"]
    steps     = dqn_results["steps"]

    success_rate = float(np.mean(successes[-EVAL_WINDOW:]))

    recent_steps = steps[-EVAL_WINDOW:]
    valid_steps  = recent_steps[~np.isnan(recent_steps)]
    avg_steps    = float(np.mean(valid_steps)) if len(valid_steps) > 0 else float("nan")

    return {"success_rate": success_rate, "avg_steps": avg_steps}


def dqn_param_count() -> int:
    """
    Count trainable parameters in DuelingDQN(8, 3).

    This number is constant regardless of the maze size — the network
    architecture does not change as the grid grows.  Compare this to
    the Q-table, which adds a new entry for every newly visited state.
    """
    net = DuelingDQN(input_dim=8, output_dim=3)
    return sum(p.numel() for p in net.parameters())


# ─── Combined scalability plot ────────────────────────────────────────────────

def plot_combined(tabular: dict, dqn_sum: dict, n_params: int,
                  filename: str = "scalability_with_dqn.png") -> None:
    """
    3-panel scalability figure with DQN overlaid on tabular results.

    Panel 1 — Success rate vs grid size
        SARSA (blue) and Q-Learning (red) curves across all tested sizes.
        DQN (green ★) plotted as a single marker at 16×16.
        Shows DQN outperforming tabular methods at the largest scale.

    Panel 2 — Model size vs grid size  (log scale)
        Q-table unique-state counts grow with the maze (curse of dimensionality).
        DQN represented as a horizontal dashed line at its fixed parameter count.
        Log scale used because the two quantities have very different magnitudes
        (hundreds of Q-states vs tens of thousands of network weights) but the
        KEY visual story — constant vs growing — is still clear.

    Panel 3 — Avg steps to goal vs grid size
        Tabular curves plotted where steps data is available.
        DQN ★ at 16×16 if any successful episodes were recorded.
    """
    grid_sizes    = tabular["grid_sizes"]
    sarsa_summary = tabular["sarsa_summary"]
    ql_summary    = tabular["ql_summary"]
    n_eps         = tabular["n_episodes"]
    eval_w        = tabular["eval_window"]

    xs = np.array(grid_sizes)
    mk = "o"

    sarsa_sr    = [sarsa_summary[g]["success_rate"] for g in grid_sizes]
    ql_sr       = [ql_summary[g]["success_rate"]    for g in grid_sizes]
    sarsa_qs    = [sarsa_summary[g]["q_table_size"] for g in grid_sizes]
    ql_qs       = [ql_summary[g]["q_table_size"]    for g in grid_sizes]
    sarsa_steps = [sarsa_summary[g]["avg_steps"]    for g in grid_sizes]
    ql_steps    = [ql_summary[g]["avg_steps"]       for g in grid_sizes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Scalability: Tabular Methods vs Improved DQN  |  "
        f"Tabular: {n_eps} eps/size  |  DQN: {DQN_GRID_SIZE}×{DQN_GRID_SIZE}  "
        f"(Double DQN + Dueling + Soft updates)",
        fontsize=11, fontweight="bold"
    )

    tick_labels = [f"{g}×{g}" for g in grid_sizes]

    # ── Panel 1: Success rate ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(xs, sarsa_sr,
            color=SARSA_COLOR, marker=mk, lw=2, ms=8, label="SARSA")
    ax.plot(xs, ql_sr,
            color=QL_COLOR, marker=mk, lw=2, ms=8, label="Q-Learning")
    ax.plot(DQN_GRID_SIZE, dqn_sum["success_rate"],
            color=DQN_COLOR, marker="*", markersize=18, linestyle="none",
            label="DQN (improved)", zorder=5)
    ax.set_title(f"Final Success Rate vs Maze Size\n(last {eval_w} episodes)")
    ax.set_xlabel("Grid Size (N×N)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Model size (log scale) ─────────────────────────────────────
    ax = axes[1]
    ax.plot(xs, sarsa_qs,
            color=SARSA_COLOR, marker=mk, lw=2, ms=8,
            label="SARSA (unique Q-states)")
    ax.plot(xs, ql_qs,
            color=QL_COLOR, marker=mk, lw=2, ms=8,
            label="Q-Lrn (unique Q-states)")
    ax.axhline(y=n_params, color=DQN_COLOR, linestyle="--", lw=2,
               label=f"DQN weights = {n_params:,}  (constant)")
    ax.set_yscale("log")
    ax.set_title("Model Size vs Maze Size\n(Q-states grow; DQN weights constant)")
    ax.set_xlabel("Grid Size (N×N)")
    ax.set_ylabel("Count  (log scale)")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 3: Avg steps to goal ───────────────────────────────────────────
    ax = axes[2]
    for steps, color, label in [
        (sarsa_steps, SARSA_COLOR, "SARSA"),
        (ql_steps,    QL_COLOR,    "Q-Learning"),
    ]:
        vx = [xs[i] for i, v in enumerate(steps) if not np.isnan(v)]
        vy = [v      for v in steps               if not np.isnan(v)]
        if vy:
            ax.plot(vx, vy, color=color, marker=mk, lw=2, ms=8, label=label)
    if not np.isnan(dqn_sum["avg_steps"]):
        ax.plot(DQN_GRID_SIZE, dqn_sum["avg_steps"],
                color=DQN_COLOR, marker="*", markersize=18, linestyle="none",
                label="DQN (improved)", zorder=5)
    ax.set_title(f"Avg Steps to Goal vs Maze Size\n(last {eval_w} successful eps)")
    ax.set_xlabel("Grid Size (N×N)")
    ax.set_ylabel("Steps")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {filename}")


# ─── Console summary table ────────────────────────────────────────────────────

def print_table(tabular: dict, dqn_sum: dict, n_params: int) -> None:
    grid_sizes    = tabular["grid_sizes"]
    sarsa_summary = tabular["sarsa_summary"]
    ql_summary    = tabular["ql_summary"]

    print("\n" + "=" * 82)
    print("  COMBINED SCALABILITY SUMMARY  (tabular methods + improved DQN)")
    print("=" * 82)
    print(f"  {'Grid':>6}  {'SARSA SR':>9}  {'QL SR':>9}  {'DQN SR':>9}  "
          f"{'SARSA states':>13}  {'QL states':>10}  {'DQN weights':>12}")
    print("  " + "─" * 76)
    for g in grid_sizes:
        dqn_sr  = f"{dqn_sum['success_rate']:.1%}" if g == DQN_GRID_SIZE else "—"
        dqn_sz  = f"{n_params:,}"                  if g == DQN_GRID_SIZE else "—"
        print(f"  {g}×{g:<3}  "
              f"{sarsa_summary[g]['success_rate']:>9.1%}  "
              f"{ql_summary[g]['success_rate']:>9.1%}  "
              f"{dqn_sr:>9}  "
              f"{sarsa_summary[g]['q_table_size']:>13,}  "
              f"{ql_summary[g]['q_table_size']:>10,}  "
              f"{dqn_sz:>12}")
    print("=" * 82)
    steps_str = "n/a" if np.isnan(dqn_sum["avg_steps"]) else f"{dqn_sum['avg_steps']:.1f}"
    print(f"\n  DQN avg steps to goal at 16×16  : {steps_str}")
    print(f"  DQN parameter count (constant)  : {n_params:,}")
    print(f"\n  Key takeaway: at 16×16, DQN achieves {dqn_sum['success_rate']:.1%} success")
    print(f"  vs SARSA {tabular['sarsa_summary'][16]['success_rate']:.1%}"
          f" / Q-Learning {tabular['ql_summary'][16]['success_rate']:.1%}.")
    print("  DQN's parameter count does not grow with maze size.")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  DQN vs Tabular Scalability Comparison")
    print("=" * 55)

    print("\nLoading tabular data ...")
    tabular = load_tabular_data()

    print("Loading DQN results ...")
    dqn_results = load_dqn_results()

    dqn_sum  = compute_dqn_summary(dqn_results)
    n_params = dqn_param_count()

    print(f"  DQN success rate (last {EVAL_WINDOW} eps) : {dqn_sum['success_rate']:.1%}")
    steps_str = "n/a" if np.isnan(dqn_sum["avg_steps"]) else f"{dqn_sum['avg_steps']:.1f}"
    print(f"  DQN avg steps to goal            : {steps_str}")
    print(f"  DQN parameter count              : {n_params:,}")

    print("\nGenerating plot ...")
    plot_combined(tabular, dqn_sum, n_params)

    print_table(tabular, dqn_sum, n_params)

    print("\nDone.")
