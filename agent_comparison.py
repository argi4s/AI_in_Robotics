import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.simple_dungeonworld_env import DungeonMazeEnv
from models.sarsa import SARSAAgent
from models.q_learning import QLearningAgent
from utils.state_encoder import encode_state


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────

# Maze sizes to sweep. All values must be even and >= 6.
GRID_SIZES = [8, 10, 12, 14, 16]

# Fixed training budget for every size. Keeping it constant makes the
# scalability comparison fair: "given the same number of episodes, how
# does each agent perform as the maze grows?"
N_EPISODES = 1500

# Shared hyperparameters (identical for both agents at every grid size)
ALPHA         = 0.1
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.999   # per-episode multiplicative decay
EPSILON_MIN   = 0.01

ROLLING_WIN  = 50    # window size for smoothing training curves
EVAL_WINDOW  = 200   # episodes used when computing final summary statistics
SHARED_SEED  = 42    # master seed — ensures both agents see identical mazes

# Plot colours
SARSA_COLOR = "#2196F3"   # blue
QL_COLOR    = "#F44336"   # red


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (shared by both agent types)
# ─────────────────────────────────────────────────────────────────────────────

def train(agent_type: str, grid_size: int, n_episodes: int, seed: int) -> dict:
    """
    Train one agent on a maze of the given size and return per-episode metrics.

    The seed drives a reproducible sequence of distinct maze layouts so that
    SARSA and Q-Learning always see the exact same mazes in the same order.

    Parameters
    ----------
    agent_type : "sarsa" | "qlearning"
    grid_size  : width (= height) of the square maze grid
    n_episodes : total training episodes
    seed       : master RNG seed for the maze sequence

    Returns
    -------
    dict with per-episode lists:
        rewards    — total undiscounted reward
        steps      — steps taken (None when episode failed)
        successes  — 1 if goal reached, 0 otherwise
        q_sizes    — number of unique states in the Q-table
    """
    env = DungeonMazeEnv(grid_size=grid_size)

    if agent_type == "sarsa":
        agent = SARSAAgent(action_size=env.action_space.n)
    else:
        agent = QLearningAgent(action_size=env.action_space.n)

    epsilon = EPSILON_START
    rng     = np.random.default_rng(seed)

    metrics     = {"rewards": [], "steps": [], "successes": [], "q_sizes": []}
    best_reward = -float("inf")
    best_Q      = None

    for episode in range(n_episodes):
        # Derive a per-episode seed from the shared RNG so that both agents
        # visit identical maze layouts in the same order.
        ep_seed = int(rng.integers(0, 2**31))
        obs, _  = env.reset(seed=ep_seed)
        state   = encode_state(obs)

        # SARSA is on-policy: the next action must be chosen before stepping.
        if agent_type == "sarsa":
            action = agent.select_action(state, epsilon)

        total_reward = 0
        steps        = 0
        success      = False

        while True:
            if agent_type == "sarsa":
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state  = encode_state(next_obs)
                next_action = agent.select_action(next_state, epsilon)
                agent.update(state, action, reward, next_state, next_action,
                             ALPHA, GAMMA)
                action = next_action
            else:
                # Q-Learning is off-policy: action chosen greedily from current Q.
                action = agent.select_action(state, epsilon)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = encode_state(next_obs)
                agent.update(state, action, reward, next_state, ALPHA, GAMMA)

            state        = next_state
            total_reward += reward
            steps        += 1

            if terminated:
                success = True
            if terminated or truncated:
                break

        # Decay exploration rate
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Track the best Q-table seen so far
        if total_reward > best_reward:
            best_reward = total_reward
            best_Q      = copy.deepcopy(agent.Q)

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(steps if success else None)
        metrics["successes"].append(int(success))
        metrics["q_sizes"].append(len(agent.Q))

        print(f"  [{agent_type.upper():>10}] {grid_size}x{grid_size} "
              f"ep {episode+1:4d}/{n_episodes} | "
              f"R {total_reward:7.2f} | "
              f"{'✓' if success else '✗'} | "
              f"ε {epsilon:.3f} | "
              f"Q {len(agent.Q):5d}")

    # Persist the best Q-table found during training (one file per grid size)
    fname = (f"sarsa_table_{grid_size}.pkl" if agent_type == "sarsa"
             else f"q_table_{grid_size}.pkl")
    with open(fname, "wb") as fh:
        pickle.dump(best_Q, fh)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mean(data: list, window: int) -> list:
    """Causal rolling average (no look-ahead). Window shrinks at the start."""
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(data[lo:i + 1])))
    return out


def steps_per_success(steps_list: list):
    """
    Split steps_list into (episode_indices, step_counts) keeping only
    episodes where the agent reached the goal.
    """
    xs, ys = [], []
    for i, s in enumerate(steps_list):
        if s is not None:
            xs.append(i)
            ys.append(s)
    return xs, ys


def final_success_rate(successes: list, window: int = EVAL_WINDOW) -> float:
    """Mean success rate over the last `window` episodes."""
    return float(np.mean(successes[-window:]))


def final_avg_steps(steps_list: list, window: int = EVAL_WINDOW) -> float:
    """
    Mean steps-to-goal over the last `window` successful episodes.
    Returns nan if no successful episodes exist in that window.
    """
    recent = [s for s in steps_list[-window:] if s is not None]
    return float(np.mean(recent)) if recent else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Per-size comparison plot (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(grid_size: int, sarsa_m: dict, ql_m: dict,
                    n_episodes: int, filename: str) -> None:
    """
    Save a 2×2 figure comparing SARSA and Q-Learning for one grid size.

    Panels
    ------
    1. Episode reward (raw + smoothed)
    2. Rolling success rate
    3. Steps to goal (scatter + smoothed, successful episodes only)
    4. Q-table size (unique states visited over time)
    """
    episodes = np.arange(n_episodes)

    sarsa_smooth = rolling_mean(sarsa_m["rewards"],    ROLLING_WIN)
    ql_smooth    = rolling_mean(ql_m["rewards"],       ROLLING_WIN)
    sarsa_sr     = rolling_mean(sarsa_m["successes"],  ROLLING_WIN)
    ql_sr        = rolling_mean(ql_m["successes"],     ROLLING_WIN)
    sarsa_sx, sarsa_sy = steps_per_success(sarsa_m["steps"])
    ql_sx,    ql_sy    = steps_per_success(ql_m["steps"])

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"SARSA vs Q-Learning  |  grid={grid_size}×{grid_size},  "
        f"{n_episodes} episodes,  α={ALPHA},  γ={GAMMA}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    # ── Panel 1: Episode reward ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, sarsa_m["rewards"],
             color=SARSA_COLOR, alpha=0.15, linewidth=0.6)
    ax1.plot(episodes, ql_m["rewards"],
             color=QL_COLOR,    alpha=0.15, linewidth=0.6)
    ax1.plot(episodes, sarsa_smooth,
             color=SARSA_COLOR, linewidth=2, label="SARSA")
    ax1.plot(episodes, ql_smooth,
             color=QL_COLOR,    linewidth=2, label="Q-Learning")
    ax1.set_title("Episode Reward (raw + smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Success rate ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, sarsa_sr,
             color=SARSA_COLOR, linewidth=2, label="SARSA")
    ax2.plot(episodes, ql_sr,
             color=QL_COLOR,    linewidth=2, label="Q-Learning")
    ax2.set_title(f"Success Rate  (rolling {ROLLING_WIN}-ep window)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Steps to goal ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if sarsa_sy:
        sarsa_steps_sm = rolling_mean(sarsa_sy, min(ROLLING_WIN, len(sarsa_sy)))
        ax3.scatter(sarsa_sx, sarsa_sy,
                    color=SARSA_COLOR, alpha=0.2, s=8)
        ax3.plot(sarsa_sx, sarsa_steps_sm,
                 color=SARSA_COLOR, linewidth=2, label="SARSA")
    if ql_sy:
        ql_steps_sm = rolling_mean(ql_sy, min(ROLLING_WIN, len(ql_sy)))
        ax3.scatter(ql_sx, ql_sy,
                    color=QL_COLOR, alpha=0.2, s=8)
        ax3.plot(ql_sx, ql_steps_sm,
                 color=QL_COLOR, linewidth=2, label="Q-Learning")
    ax3.set_title("Steps to Goal  (successful episodes only)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Q-table size ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, sarsa_m["q_sizes"],
             color=SARSA_COLOR, linewidth=2, label="SARSA")
    ax4.plot(episodes, ql_m["q_sizes"],
             color=QL_COLOR,    linewidth=2, label="Q-Learning")
    ax4.set_title("Q-Table Size  (unique states visited)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("# Unique States")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Scalability summary plot (3 panels, one data point per grid size)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scalability(grid_sizes: list, sarsa_summary: dict,
                     ql_summary: dict, filename: str) -> None:
    """
    Three-panel figure showing how performance changes as the maze grows.

    Panels
    ------
    1. Final success rate vs grid size   — does the agent still solve the maze?
    2. Final Q-table size vs grid size   — how much memory does the table need?
    3. Avg steps to goal vs grid size    — does the agent find efficient paths?

    The growing Q-table and falling success rate together make the case for
    moving to a function-approximation method (DQN) for larger grids.
    """
    xs = np.array(grid_sizes)

    sarsa_sr    = [sarsa_summary[g]["success_rate"] for g in grid_sizes]
    ql_sr       = [ql_summary[g]["success_rate"]    for g in grid_sizes]
    sarsa_qs    = [sarsa_summary[g]["q_table_size"] for g in grid_sizes]
    ql_qs       = [ql_summary[g]["q_table_size"]    for g in grid_sizes]
    sarsa_steps = [sarsa_summary[g]["avg_steps"]    for g in grid_sizes]
    ql_steps    = [ql_summary[g]["avg_steps"]       for g in grid_sizes]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Scalability: SARSA vs Q-Learning  |  "
        f"{N_EPISODES} episodes per size,  α={ALPHA},  γ={GAMMA}",
        fontsize=13, fontweight="bold"
    )

    mk = "o"   # marker style

    # ── Panel 1: Success rate ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(xs, sarsa_sr,
            color=SARSA_COLOR, marker=mk, linewidth=2, markersize=8, label="SARSA")
    ax.plot(xs, ql_sr,
            color=QL_COLOR,    marker=mk, linewidth=2, markersize=8, label="Q-Learning")
    ax.set_title(f"Final Success Rate vs Maze Size\n(last {EVAL_WINDOW} episodes)")
    ax.set_xlabel("Grid Size (N×N)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels([f"{g}×{g}" for g in grid_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Q-table size ────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(xs, sarsa_qs,
            color=SARSA_COLOR, marker=mk, linewidth=2, markersize=8, label="SARSA")
    ax.plot(xs, ql_qs,
            color=QL_COLOR,    marker=mk, linewidth=2, markersize=8, label="Q-Learning")
    ax.set_title("Final Q-Table Size vs Maze Size\n(unique states discovered)")
    ax.set_xlabel("Grid Size (N×N)")
    ax.set_ylabel("# Unique States")
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels([f"{g}×{g}" for g in grid_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Avg steps to goal ───────────────────────────────────────────
    ax = axes[2]
    for agent_steps, color, label in [
        (sarsa_steps, SARSA_COLOR, "SARSA"),
        (ql_steps,    QL_COLOR,    "Q-Learning"),
    ]:
        # Only plot grid sizes where at least one episode succeeded
        valid_xs = [xs[i] for i, v in enumerate(agent_steps) if not np.isnan(v)]
        valid_ys = [v      for v in agent_steps               if not np.isnan(v)]
        if valid_ys:
            ax.plot(valid_xs, valid_ys,
                    color=color, marker=mk, linewidth=2, markersize=8, label=label)
    ax.set_title(f"Avg Steps to Goal vs Maze Size\n(last {EVAL_WINDOW} successful eps)")
    ax.set_xlabel("Grid Size (N×N)")
    ax.set_ylabel("Steps")
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels([f"{g}×{g}" for g in grid_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sarsa_summary: dict = {}
    ql_summary:    dict = {}

    for grid_size in GRID_SIZES:
        header = f"Grid {grid_size}×{grid_size}"

        # ── Train SARSA ──────────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print(f"  {header}  —  Training SARSA  ({N_EPISODES} episodes)")
        print("=" * 65)
        sarsa_m = train("sarsa", grid_size, N_EPISODES, SHARED_SEED)

        # ── Train Q-Learning ─────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print(f"  {header}  —  Training Q-Learning  ({N_EPISODES} episodes)")
        print("=" * 65)
        ql_m = train("qlearning", grid_size, N_EPISODES, SHARED_SEED)

        # ── Per-size plot ────────────────────────────────────────────────────
        plot_comparison(
            grid_size, sarsa_m, ql_m, N_EPISODES,
            filename=f"comparison_{grid_size}x{grid_size}.png"
        )

        # ── Collect scalability metrics ──────────────────────────────────────
        sarsa_summary[grid_size] = {
            "success_rate": final_success_rate(sarsa_m["successes"]),
            "q_table_size": sarsa_m["q_sizes"][-1],
            "avg_steps":    final_avg_steps(sarsa_m["steps"]),
        }
        ql_summary[grid_size] = {
            "success_rate": final_success_rate(ql_m["successes"]),
            "q_table_size": ql_m["q_sizes"][-1],
            "avg_steps":    final_avg_steps(ql_m["steps"]),
        }

        # ── Per-size console summary ─────────────────────────────────────────
        s  = sarsa_summary[grid_size]
        q  = ql_summary[grid_size]
        print(f"\n  {'─'*52}")
        print(f"  {header} Summary")
        print(f"  {'─'*52}")
        print(f"  {'Metric':<32} {'SARSA':>9}  {'Q-Lrn':>9}")
        print(f"  {'─'*52}")
        print(f"  {'Success rate  (last 200 ep)':<32} "
              f"{s['success_rate']:>9.1%}  {q['success_rate']:>9.1%}")
        print(f"  {'Q-table size  (final)':<32} "
              f"{s['q_table_size']:>9d}  {q['q_table_size']:>9d}")
        s_steps_str = "n/a" if np.isnan(s["avg_steps"]) else f"{s['avg_steps']:.1f}"
        q_steps_str = "n/a" if np.isnan(q["avg_steps"]) else f"{q['avg_steps']:.1f}"
        print(f"  {'Avg steps to goal':<32} "
              f"{s_steps_str:>9}  {q_steps_str:>9}")

    # ── Scalability summary plot ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generating scalability summary ...")
    print("=" * 65)
    plot_scalability(GRID_SIZES, sarsa_summary, ql_summary,
                     filename="scalability_summary.png")

    # ── Final cross-size table ───────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  SCALABILITY SUMMARY  (tabular methods vs maze size)")
    print("═" * 68)
    print(f"  {'Grid':>6}  {'SARSA SR':>9}  {'QL SR':>9}  "
          f"{'SARSA Q-states':>15}  {'QL Q-states':>12}")
    print(f"  {'─'*64}")
    for g in GRID_SIZES:
        print(f"  {g}×{g:<3}  "
              f"{sarsa_summary[g]['success_rate']:>9.1%}  "
              f"{ql_summary[g]['success_rate']:>9.1%}  "
              f"{sarsa_summary[g]['q_table_size']:>15,}  "
              f"{ql_summary[g]['q_table_size']:>12,}")
    print("═" * 68)
    print("\n  Done. Check the PNG files in the project root.")

    # ── Save summary for compare_dqn_tabular.py ──────────────────────────────
    # Avoids re-running the 30-minute tabular training when generating the
    # combined DQN vs tabular comparison plot.
    import pickle as _pickle
    _data = {
        "grid_sizes":    GRID_SIZES,
        "sarsa_summary": sarsa_summary,
        "ql_summary":    ql_summary,
        "n_episodes":    N_EPISODES,
        "eval_window":   EVAL_WINDOW,
    }
    with open("scalability_summary.pkl", "wb") as _fh:
        _pickle.dump(_data, _fh)
    print("  Saved → scalability_summary.pkl  (used by compare_dqn_tabular.py)")
