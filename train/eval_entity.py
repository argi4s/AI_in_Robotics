import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from envs.entity_dungeonworld_env import DungeonMazeEntityEnv, EntityActions, ENERGY_MAX
from models.dqn_network import DuelingDQN
from utils.obs_utils import flatten_observation_entity

RESULTS_DIR  = "results"
MODEL_PATH   = os.path.join("saved_models", "best_dqn_entity_model.pth")
GRID_SIZE    = 12
TRAIN_SEED   = 77      # seed the oracle model was trained on
EVAL_EPISODES = 200    # same window used during training
MAX_STEPS    = 300
STATE_DIM    = 17
ACTION_DIM   = len(EntityActions)

ENTITY_POSITIONS = {'tank': (6, 4), 'smart': (3, 4)}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Load trained model ───────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net  = DuelingDQN(STATE_DIM, ACTION_DIM).to(device)
q_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
q_net.eval()
print(f"Loaded model from {MODEL_PATH}  (device: {device})")


def greedy_action(state: np.ndarray) -> int:
    with torch.no_grad():
        t = torch.FloatTensor(state).unsqueeze(0).to(device)
        return int(q_net(t).argmax().item())


# ─── Generic evaluation loop ──────────────────────────────────────────────────

def evaluate(env, action_fn, n_episodes: int, seed: int):
    """
    Run n_episodes on env using action_fn.  Returns (successes, steps_list).
    seed is used as a base; each episode offsets by episode index so the maze
    layout is fixed (same as training) for the given seed.
    """
    successes  = []
    steps_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        state  = flatten_observation_entity(obs, GRID_SIZE)
        success, steps = False, 0

        for _ in range(MAX_STEPS):
            action = action_fn(state)
            obs, _, terminated, truncated, _ = env.step(action)
            state = flatten_observation_entity(obs, GRID_SIZE)
            steps += 1
            if terminated:
                success = True
            if terminated or truncated:
                break

        successes.append(int(success))
        steps_list.append(steps if success else None)

    sr    = float(np.mean(successes))
    valid = [s for s in steps_list if s is not None]
    avg_steps = float(np.mean(valid)) if valid else float("nan")
    return successes, steps_list, sr, avg_steps


# ─── 1. Random baseline ───────────────────────────────────────────────────────

print("\n[1/4] Random baseline ...")
env_oracle = DungeonMazeEntityEnv(grid_size=GRID_SIZE, use_shaping=True,
                                   entity_positions=ENTITY_POSITIONS,
                                   use_perception=False)
env_oracle.max_steps = MAX_STEPS

random_action = lambda state: env_oracle.action_space.sample()
rand_s, rand_st, rand_sr, rand_avg = evaluate(env_oracle, random_action,
                                               EVAL_EPISODES, TRAIN_SEED)
np.save(os.path.join(RESULTS_DIR, "eval_random_baseline.npy"), np.array(rand_s))
print(f"  Random baseline  | SR: {rand_sr:.1%}  | Avg steps: {rand_avg:.1f}"
      if not np.isnan(rand_avg) else f"  Random baseline  | SR: {rand_sr:.1%}  | Avg steps: n/a")


# ─── 2. Oracle model + oracle observations (training regime) ──────────────────

print("\n[2/4] Oracle model + oracle observations ...")
ora_s, ora_st, ora_sr, ora_avg = evaluate(env_oracle, greedy_action,
                                           EVAL_EPISODES, TRAIN_SEED)
np.save(os.path.join(RESULTS_DIR, "eval_oracle_oracle.npy"), np.array(ora_s))
print(f"  Oracle / oracle  | SR: {ora_sr:.1%}  | Avg steps: {ora_avg:.1f}"
      if not np.isnan(ora_avg) else f"  Oracle / oracle  | SR: {ora_sr:.1%}  | Avg steps: n/a")


# ─── 3. Oracle model + perception observations (cross-test) ───────────────────

print("\n[3/4] Oracle model + perception observations (cross-test) ...")
print("  (loading Task 1 classifiers — may take a moment)")
try:
    env_perc = DungeonMazeEntityEnv(grid_size=GRID_SIZE, use_shaping=True,
                                     entity_positions=ENTITY_POSITIONS,
                                     use_perception=True)
    env_perc.max_steps = MAX_STEPS
    perc_s, perc_st, perc_sr, perc_avg = evaluate(env_perc, greedy_action,
                                                    EVAL_EPISODES, TRAIN_SEED)
    np.save(os.path.join(RESULTS_DIR, "eval_oracle_perception.npy"), np.array(perc_s))
    perc_ok = True
    print(f"  Oracle / percept | SR: {perc_sr:.1%}  | Avg steps: {perc_avg:.1f}"
          if not np.isnan(perc_avg) else
          f"  Oracle / percept | SR: {perc_sr:.1%}  | Avg steps: n/a")
except Exception as e:
    print(f"  Perception unavailable ({e}) — skipping cross-test.")
    perc_sr, perc_avg, perc_ok = float("nan"), float("nan"), False


# ─── 4. Generalisation test — unseen maze seeds ────────────────────────────────

print("\n[4/4] Generalisation test (unseen seeds) ...")
UNSEEN_SEEDS = [1234, 5678, 9999]
gen_srs, gen_avgs = [], []

for useed in UNSEEN_SEEDS:
    g_s, g_st, g_sr, g_avg = evaluate(env_oracle, greedy_action,
                                        EVAL_EPISODES, useed)
    gen_srs.append(g_sr)
    gen_avgs.append(g_avg)
    print(f"  Seed {useed}  | SR: {g_sr:.1%}  | Avg steps: "
          + (f"{g_avg:.1f}" if not np.isnan(g_avg) else "n/a"))

gen_sr_mean = float(np.mean(gen_srs))
gen_sr_std  = float(np.std(gen_srs))
np.save(os.path.join(RESULTS_DIR, "eval_generalisation.npy"), np.array(gen_srs))
print(f"\n  Generalisation   | SR: {gen_sr_mean:.1%} ± {gen_sr_std:.1%}")


# ─── Summary table ────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  EVALUATION SUMMARY")
print("=" * 62)
print(f"  {'Configuration':<30} {'Success Rate':>13}  {'Avg Steps':>10}")
print(f"  {'-'*58}")
print(f"  {'Random baseline':<30} {rand_sr:>13.1%}  {'n/a' if np.isnan(rand_avg) else f'{rand_avg:.1f}':>10}")
print(f"  {'Oracle model (oracle obs)':<30} {ora_sr:>13.1%}  {'n/a' if np.isnan(ora_avg) else f'{ora_avg:.1f}':>10}")
if perc_ok:
    print(f"  {'Oracle model (percept obs)':<30} {perc_sr:>13.1%}  {'n/a' if np.isnan(perc_avg) else f'{perc_avg:.1f}':>10}")
print(f"  {'Generalisation (mean±std)':<30} {gen_sr_mean:>12.1%}  {'':>10}")
for i, s in enumerate(UNSEEN_SEEDS):
    print(f"    seed {s:<6}                       {gen_srs[i]:>12.1%}")
print("=" * 62)


# ─── Plot ─────────────────────────────────────────────────────────────────────

labels = ["Random\nbaseline", "Oracle\n(oracle obs)", "Oracle\n(percept obs)",
          f"Generalise\n(mean, n={len(UNSEEN_SEEDS)})"]
values = [rand_sr, ora_sr,
          perc_sr if perc_ok else float("nan"),
          gen_sr_mean]
errors = [0, 0,
          0,
          gen_sr_std]

colours = ["#888888", "#2196F3", "#FF9800", "#4CAF50"]

fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(len(labels))
bars = ax.bar(xs, values, yerr=errors, capsize=6, color=colours,
              edgecolor="white", width=0.55, alpha=0.85)

for bar, v in zip(bars, values):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Success Rate", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("Entity DQN Evaluation Summary\n"
             "(oracle model, 200 eval episodes, 12×12 fixed maze)",
             fontsize=12, fontweight="bold")
ax.axhline(rand_sr, color="#888888", linestyle="--", alpha=0.5, linewidth=1)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eval_entity_summary.png"), dpi=150)
plt.close()
print(f"\nSaved -> {RESULTS_DIR}/eval_entity_summary.png")
print("Done.")
