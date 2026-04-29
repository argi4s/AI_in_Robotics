import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions
from utils.obs_utils import flatten_observation_v2
from models.dqn_network import DuelingDQN
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
GRID_SIZE    = 12
MAZE_SEED    = 42      # fixed maze — same layout every episode
EPISODES     = 3000    # extended budget — agent was still learning at 1500
MAX_STEPS    = 300     # overrides env's hardcoded 200-step cap post-construction
BATCH_SIZE   = 128     # larger batches → better gradient estimates
WARMUP_STEPS = 500     # fill buffer with diverse transitions before first update

# Target network update — hard copy every N episodes.
# Polyak (tau=0.005 per gradient step) was tried but caused instability:
# with 300 grad steps/episode the target was 78% converged to the online
# network after just 1 episode, effectively removing the stabilising lag.
TARGET_UPDATE_FREQ = 50   # hard copy every 50 episodes

# Agent hyperparameters
GAMMA         = 0.99
LR            = 1e-3
EPSILON_START = 1.0
EPSILON_MIN   = 0.05   # lower floor — lets the agent exploit what it has
                        # learned in later episodes rather than staying noisy
EPSILON_DECAY = 0.998  # slower decay — ε reaches EPSILON_MIN around episode 2300
                        # giving more exploration time over the longer run

TAU = 1.0              # tells DQNAgent.update_target() to do a hard copy

SMOOTH_WIN   = 20      # moving-average window for training plot
EVAL_WINDOW  = 200     # last N episodes for final summary statistics


# ─── Environment ──────────────────────────────────────────────────────────────
env = DungeonMazeEnv(grid_size=GRID_SIZE, use_shaping=False)
env.max_steps = MAX_STEPS

obs, _    = env.reset(seed=MAZE_SEED)
state     = flatten_observation_v2(obs, GRID_SIZE)
state_dim  = state.shape[0]   # 10
action_dim = len(Actions)     # 3


# ─── Networks, Agent, Replay Buffer ───────────────────────────────────────────
q_net      = DuelingDQN(state_dim, action_dim)
target_net = DuelingDQN(state_dim, action_dim)

agent = DQNAgent(
    state_dim, action_dim,
    gamma         = GAMMA,
    lr            = LR,
    epsilon_start = EPSILON_START,
    epsilon_min   = EPSILON_MIN,
    epsilon_decay = EPSILON_DECAY,
    use_double    = True,   # Double DQN
    tau           = TAU,    # soft Polyak updates
)
agent.set_networks(q_net, target_net)

buffer = ReplayBuffer(capacity=100_000)


# ─── Metric tracking ──────────────────────────────────────────────────────────
episode_rewards   = []
episode_losses    = []
episode_successes = []   # float 1.0 / 0.0
episode_steps     = []   # float steps, NaN for failed episodes
best_reward       = -float("inf")

print(f"Training DQN on {GRID_SIZE}×{GRID_SIZE} maze for {EPISODES} episodes.")
print(f"State dim: {state_dim}  |  Actions: {action_dim}  |  Device: {agent.device}")
print(f"Architecture: DuelingDQN  |  Double DQN: {agent.use_double}  |  Hard update every {TARGET_UPDATE_FREQ} eps")
print("─" * 65)


# ─── Training loop ────────────────────────────────────────────────────────────
for episode in range(EPISODES):
    obs, _  = env.reset(seed=MAZE_SEED)
    state   = flatten_observation_v2(obs, GRID_SIZE)

    total_reward = 0.0
    ep_losses    = []
    success      = False
    steps_taken  = 0

    for step in range(MAX_STEPS):
        action_idx = agent.act(state)

        obs, reward, terminated, truncated, _ = env.step(action_idx)
        next_state = flatten_observation_v2(obs, GRID_SIZE)
        done = terminated or truncated

        buffer.push(state, action_idx, reward, next_state, done)
        state        = next_state
        total_reward += reward
        steps_taken  += 1

        # Gradient update after warmup
        if len(buffer) > WARMUP_STEPS:
            batch = buffer.sample(BATCH_SIZE)
            loss  = agent.update(batch)
            ep_losses.append(loss)

        if terminated:
            success = True
        if done:
            break

    agent.decay_epsilon()   # once per episode

    # Hard target network update every TARGET_UPDATE_FREQ episodes.
    # This keeps the target fixed long enough to be a stable learning signal.
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target()

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.q_net.state_dict(), os.path.join(RESULTS_DIR, "best_dqn_model.pth"))

    episode_rewards.append(total_reward)
    episode_losses.append(np.mean(ep_losses) if ep_losses else float("nan"))
    episode_successes.append(1.0 if success else 0.0)
    episode_steps.append(float(steps_taken) if success else float("nan"))

    print(
        f"Ep {episode + 1:4d}/{EPISODES} | "
        f"R {total_reward:8.2f} | "
        f"{'WIN' if success else '   '} | "
        f"Steps {steps_taken:3d} | "
        f"Loss {episode_losses[-1]:.4f} | "
        f"ε {agent.epsilon:.3f}"
    )


# ─── Save metrics ─────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "dqn_successes.npy"), np.array(episode_successes))
np.save(os.path.join(RESULTS_DIR, "dqn_steps.npy"),     np.array(episode_steps))
print(f"\nSaved → {RESULTS_DIR}/dqn_successes.npy, dqn_steps.npy")
print(f"Saved → {RESULTS_DIR}/best_dqn_model.pth")


# ─── Final summary ────────────────────────────────────────────────────────────
final_sr    = float(np.mean(episode_successes[-EVAL_WINDOW:]))
valid_steps = [s for s in episode_steps[-EVAL_WINDOW:] if not np.isnan(s)]
final_steps = float(np.mean(valid_steps)) if valid_steps else float("nan")

print(f"\n{'─'*55}")
print(f"  DQN 16×16 — Final Training Summary")
print(f"{'─'*55}")
print(f"  Success rate  (last {EVAL_WINDOW} ep) : {final_sr:.1%}")
steps_str = "n/a" if np.isnan(final_steps) else f"{final_steps:.1f}"
print(f"  Avg steps     (last {EVAL_WINDOW} ep) : {steps_str}")
print(f"  Best reward                      : {best_reward:.2f}")
print(f"{'─'*55}")
print("\nRun  python -m compare_dqn_tabular  to generate the comparison plot.")


# ─── Training curves plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle(
    f"DQN Training  |  {GRID_SIZE}×{GRID_SIZE} maze,  {EPISODES} episodes  |  "
    f"Double DQN + Dueling + Hard target update every {TARGET_UPDATE_FREQ} eps",
    fontsize=11, fontweight="bold"
)
xs = np.arange(EPISODES)


def smooth(data, w):
    return np.convolve(data, np.ones(w) / w, mode="valid")


# Panel 1 — Episode reward
ax = axes[0]
ax.plot(xs, episode_rewards, alpha=0.25, color="steelblue", linewidth=0.8)
if EPISODES >= SMOOTH_WIN:
    ax.plot(range(SMOOTH_WIN - 1, EPISODES), smooth(episode_rewards, SMOOTH_WIN),
            color="steelblue", linewidth=2, label=f"{SMOOTH_WIN}-ep avg")
ax.set_ylabel("Total Reward")
ax.set_title("Episode Reward")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2 — Rolling success rate
sr_smooth = []
for i in range(EPISODES):
    lo = max(0, i - SMOOTH_WIN + 1)
    sr_smooth.append(float(np.mean(episode_successes[lo:i + 1])))
ax = axes[1]
ax.plot(xs, sr_smooth, color="green", linewidth=2)
ax.set_ylabel("Success Rate")
ax.set_title(f"Rolling Success Rate  ({SMOOTH_WIN}-ep window)")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Panel 3 — Mean loss (NaN-safe)
ax = axes[2]
ax.plot(xs, episode_losses, alpha=0.3, color="tomato", linewidth=0.8)
losses_filled = [l if not np.isnan(l) else 0.0 for l in episode_losses]
if EPISODES >= SMOOTH_WIN:
    ax.plot(range(SMOOTH_WIN - 1, EPISODES), smooth(losses_filled, SMOOTH_WIN),
            color="tomato", linewidth=2)
ax.set_ylabel("Mean Loss")
ax.set_xlabel("Episode")
ax.set_title("Training Loss")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dqn_training.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {RESULTS_DIR}/dqn_training.png")
plt.show()
