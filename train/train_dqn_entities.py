import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from envs.entity_dungeonworld_env import DungeonMazeEntityEnv, EntityActions
from utils.obs_utils import flatten_observation_entity
from models.dqn_network import DuelingDQN
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer

RESULTS_DIR   = "results"
SAVED_MODELS  = "saved_models"
os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(SAVED_MODELS, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
GRID_SIZE    = 12
MAZE_SEED    = 77
EPISODES     = 5000
MAX_STEPS    = 300     # generous for 12×12 with entity detours
BATCH_SIZE   = 128
WARMUP_STEPS = 500

GAMMA         = 0.99
LR            = 1e-3
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.997

# Soft Polyak target update — called every gradient step.
# Hard copies (TAU=1.0 every 50 eps) caused Q-value crashes: the target
# suddenly jumped 6 times before ep 300, destabilising the values the
# agent was building during its win streak.
TAU = 0.005
SMOOTH_WIN = 20
EVAL_WINDOW = 200


# ─── Entity spawn positions ───────────────────────────────────────────────────
# Set to None to use automatic placement (sorted walkable quarters).
# Specify (x, y) per entity type to force corner spawns — positions where a
# wall blocks the 2-block sensor until the robot rounds the turn.
# Use  python -m manual.manual_entity  to explore the maze and find good spots.
ENTITY_POSITIONS = {'tank': (6, 4), 'smart': (3, 4)}   # flying auto-placed

# ─── Environment ──────────────────────────────────────────────────────────────
env = DungeonMazeEntityEnv(grid_size=GRID_SIZE, use_shaping=True,
                           entity_positions=ENTITY_POSITIONS)
# use_shaping=True: re-enabled here because with entities blocking corridors
# the agent needs a denser navigation signal. Fixed maze removes the
# generalisation concern that caused issues in the random-maze experiments.
env.max_steps = MAX_STEPS

obs, _    = env.reset(seed=MAZE_SEED)
state     = flatten_observation_entity(obs, GRID_SIZE)
state_dim  = state.shape[0]   # 17
action_dim = len(EntityActions)  # 6


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
    use_double    = True,
    tau           = TAU,
)
agent.set_networks(q_net, target_net)

buffer = ReplayBuffer(capacity=100_000)


# ─── Metric tracking ──────────────────────────────────────────────────────────
episode_rewards   = []
episode_losses    = []
episode_successes = []
episode_steps     = []
best_reward       = -float("inf")

# Extended diagnostics
episode_energies  = []   # energy remaining when episode ends
episode_nav_r     = []   # navigation reward component per episode
episode_combat_r  = []   # combat reward component per episode
episode_goal_r    = []   # goal bonus received (0 if episode failed)
episode_wins      = []   # combat wins per episode
episode_losses_c  = []   # combat losses per episode
episode_wasted    = []   # wasted combat actions per episode
episode_end_dist  = []   # Euclidean distance to goal at episode end (cells)
cumul_steps_log   = []   # total env steps at each episode boundary

total_env_steps   = 0
COMBAT_ACTIONS    = {3, 4, 5}

print(f"Training DQN+Entities on {GRID_SIZE}×{GRID_SIZE} fixed maze "
      f"(seed={MAZE_SEED}) for {EPISODES} episodes.")
print(f"State dim: {state_dim}  |  Actions: {action_dim}  |  Device: {agent.device}")
print("─" * 65)


# ─── Training loop ────────────────────────────────────────────────────────────
for episode in range(EPISODES):
    obs, _  = env.reset(seed=MAZE_SEED)
    state   = flatten_observation_entity(obs, GRID_SIZE)

    total_reward = 0.0
    ep_losses    = []
    success      = False
    steps_taken  = 0
    nav_r_ep     = 0.0
    combat_r_ep  = 0.0
    goal_r_ep    = 0.0
    n_wins       = 0
    n_losses_c   = 0
    n_wasted     = 0

    for step in range(MAX_STEPS):
        action_idx = agent.act(state)

        obs, reward, terminated, truncated, _ = env.step(action_idx)
        next_state = flatten_observation_entity(obs, GRID_SIZE)
        done = terminated or truncated

        buffer.push(state, action_idx, reward, next_state, done)
        state        = next_state
        total_reward += reward
        steps_taken  += 1
        total_env_steps += 1

        # Reward decomposition
        if terminated:
            goal_r_ep += reward          # +100 + energy bonus
        elif action_idx in COMBAT_ACTIONS:
            combat_r_ep += reward
            if reward >= 20.0:   n_wins     += 1
            elif reward <= -3.0: n_losses_c += 1
            elif reward == -0.5: n_wasted   += 1
        else:
            nav_r_ep += reward

        if len(buffer) > WARMUP_STEPS:
            batch = buffer.sample(BATCH_SIZE)
            loss  = agent.update(batch)
            agent.update_target()   # soft Polyak every gradient step
            ep_losses.append(loss)

        if terminated:
            success = True
        if done:
            break

    agent.decay_epsilon()

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.q_net.state_dict(),
                   os.path.join(SAVED_MODELS, "best_dqn_entity_model.pth"))

    # Distance to goal from final state (dims 6-7 are relative offsets, normalised)
    end_dist = float(np.sqrt(next_state[6]**2 + next_state[7]**2) * GRID_SIZE)

    episode_rewards.append(total_reward)
    episode_losses.append(np.mean(ep_losses) if ep_losses else float("nan"))
    episode_successes.append(1.0 if success else 0.0)
    episode_steps.append(float(steps_taken) if success else float("nan"))
    episode_energies.append(float(env.energy))
    episode_nav_r.append(nav_r_ep)
    episode_combat_r.append(combat_r_ep)
    episode_goal_r.append(goal_r_ep)
    episode_wins.append(n_wins)
    episode_losses_c.append(n_losses_c)
    episode_wasted.append(n_wasted)
    episode_end_dist.append(end_dist)
    cumul_steps_log.append(total_env_steps)

    print(
        f"Ep {episode + 1:4d}/{EPISODES} | "
        f"R {total_reward:7.1f} | "
        f"{'WIN' if success else '   '} | "
        f"Steps {steps_taken:3d} | "
        f"E {env.energy:5.1f} | "
        f"W/L/W {n_wins}/{n_losses_c}/{n_wasted} | "
        f"ε {agent.epsilon:.3f}"
    )


# ─── Save metrics ─────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "entity_successes.npy"),  np.array(episode_successes))
np.save(os.path.join(RESULTS_DIR, "entity_steps.npy"),      np.array(episode_steps))
np.save(os.path.join(RESULTS_DIR, "entity_energies.npy"),   np.array(episode_energies))
np.save(os.path.join(RESULTS_DIR, "entity_nav_r.npy"),      np.array(episode_nav_r))
np.save(os.path.join(RESULTS_DIR, "entity_combat_r.npy"),   np.array(episode_combat_r))
np.save(os.path.join(RESULTS_DIR, "entity_goal_r.npy"),     np.array(episode_goal_r))
np.save(os.path.join(RESULTS_DIR, "entity_wins.npy"),       np.array(episode_wins))
np.save(os.path.join(RESULTS_DIR, "entity_losses_c.npy"),   np.array(episode_losses_c))
np.save(os.path.join(RESULTS_DIR, "entity_wasted.npy"),     np.array(episode_wasted))
np.save(os.path.join(RESULTS_DIR, "entity_end_dist.npy"),   np.array(episode_end_dist))
np.save(os.path.join(RESULTS_DIR, "entity_cumul_steps.npy"),np.array(cumul_steps_log))
print(f"\nSaved → {RESULTS_DIR}/entity_successes.npy + extended diagnostics")
print(f"Saved → {SAVED_MODELS}/best_dqn_entity_model.pth")


# ─── Final summary ────────────────────────────────────────────────────────────
final_sr    = float(np.mean(episode_successes[-EVAL_WINDOW:]))
valid_steps = [s for s in episode_steps[-EVAL_WINDOW:] if not np.isnan(s)]
final_steps = float(np.mean(valid_steps)) if valid_steps else float("nan")

print(f"\n{'─'*55}")
print(f"  DQN+Entities — Final Training Summary")
print(f"{'─'*55}")
print(f"  Success rate  (last {EVAL_WINDOW} ep) : {final_sr:.1%}")
steps_str = "n/a" if np.isnan(final_steps) else f"{final_steps:.1f}"
print(f"  Avg steps     (last {EVAL_WINDOW} ep) : {steps_str}")
print(f"  Best reward                      : {best_reward:.2f}")
print(f"{'─'*55}")


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def smooth(data, w):
    return np.convolve(data, np.ones(w) / w, mode="valid")

def rolling(data, w):
    out = []
    for i in range(len(data)):
        lo = max(0, i - w + 1)
        out.append(float(np.mean(data[lo:i + 1])))
    return out

xs = np.arange(EPISODES)
sw = SMOOTH_WIN


# ─── Figure 1: Core training curves (reward / success / loss) ─────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle(
    f"DQN+Entities  |  {GRID_SIZE}×{GRID_SIZE} fixed maze, {EPISODES} episodes  |  "
    f"Double DQN + Dueling + Soft target (τ={TAU})  |  Tank→flee  Flying→bow  Smart→sword",
    fontsize=10, fontweight="bold"
)

ax = axes[0]
ax.plot(xs, episode_rewards, alpha=0.2, color="steelblue", linewidth=0.7)
if EPISODES >= sw:
    ax.plot(range(sw - 1, EPISODES), smooth(episode_rewards, sw),
            color="steelblue", linewidth=2, label=f"{sw}-ep avg")
ax.set_ylabel("Total Reward")
ax.set_title("Episode Reward")
ax.legend(); ax.grid(True, alpha=0.3)

sr_smooth = rolling(episode_successes, sw)
ax = axes[1]
ax.plot(xs, sr_smooth, color="green", linewidth=2)
ax.set_ylabel("Success Rate")
ax.set_title(f"Rolling Success Rate  ({sw}-ep window)")
ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(xs, episode_losses, alpha=0.25, color="tomato", linewidth=0.7)
losses_filled = [l if not np.isnan(l) else 0.0 for l in episode_losses]
if EPISODES >= sw:
    ax.plot(range(sw - 1, EPISODES), smooth(losses_filled, sw),
            color="tomato", linewidth=2)
ax.set_ylabel("Mean Loss"); ax.set_xlabel("Episode")
ax.set_title("Training Loss"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "entity_training.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {RESULTS_DIR}/entity_training.png")


# ─── Figure 2: Diagnostics (6 panels) ────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(
    f"DQN+Entities — Training Diagnostics  |  {GRID_SIZE}×{GRID_SIZE}, {EPISODES} episodes",
    fontsize=12, fontweight="bold"
)

# Panel 1: Energy remaining at episode end
ax = axes[0, 0]
ax.plot(xs, episode_energies, alpha=0.2, color="orange", linewidth=0.7)
ax.plot(range(sw - 1, EPISODES), smooth(episode_energies, sw),
        color="orange", linewidth=2, label=f"{sw}-ep avg")
ax.axhline(100, color="grey", linestyle="--", alpha=0.5, label="ENERGY_MAX")
ax.set_ylabel("Energy Remaining")
ax.set_title("Energy at Episode End\n(higher = more efficient)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2: Reward breakdown (nav / combat / goal)
ax = axes[0, 1]
nav_sm    = smooth(episode_nav_r,    sw)
combat_sm = smooth(episode_combat_r, sw)
goal_sm   = smooth(episode_goal_r,   sw)
xsm = range(sw - 1, EPISODES)
ax.plot(xsm, nav_sm,    color="steelblue", linewidth=2, label="Navigation")
ax.plot(xsm, combat_sm, color="#E67E22",   linewidth=2, label="Combat")
ax.plot(xsm, goal_sm,   color="green",     linewidth=2, label="Goal bonus")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Reward Component")
ax.set_title("Reward Breakdown\n(smoothed per component)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 3: Combat outcomes per episode
ax = axes[1, 0]
ax.plot(xs, rolling(episode_wins,     sw), color="green",  linewidth=2, label="Wins")
ax.plot(xs, rolling(episode_losses_c, sw), color="red",    linewidth=2, label="Losses")
ax.plot(xs, rolling(episode_wasted,   sw), color="grey",   linewidth=1.5,
        linestyle="--", label="Wasted")
ax.set_ylabel("Count per Episode")
ax.set_title(f"Combat Outcomes  ({sw}-ep rolling avg)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 4: Combat win rate (wins / total combat attempts)
ax = axes[1, 1]
total_combat = np.array(episode_wins) + np.array(episode_losses_c) + np.array(episode_wasted)
win_rate = np.where(total_combat > 0,
                    np.array(episode_wins) / total_combat,
                    np.nan)
win_rate_filled = np.where(np.isnan(win_rate), 0.0, win_rate)
ax.plot(xs, rolling(win_rate_filled.tolist(), sw),
        color="purple", linewidth=2)
ax.set_ylabel("Win Rate")
ax.set_ylim(-0.05, 1.05)
ax.set_title(f"Combat Win Rate  ({sw}-ep rolling avg)\n"
             "(correct weapon chosen / total combat actions)")
ax.grid(True, alpha=0.3)

# Panel 5: Distance to goal at episode end
ax = axes[2, 0]
ax.plot(xs, episode_end_dist, alpha=0.2, color="mediumpurple", linewidth=0.7)
ax.plot(range(sw - 1, EPISODES), smooth(episode_end_dist, sw),
        color="mediumpurple", linewidth=2, label=f"{sw}-ep avg")
ax.set_ylabel("Distance (cells)")
ax.set_xlabel("Episode")
ax.set_title("Distance to Goal at Episode End\n(0 = goal reached)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 6: Sample efficiency — success rate vs cumulative env steps
ax = axes[2, 1]
cumul = np.array(cumul_steps_log)
sr_arr = np.array(episode_successes)
sr_roll = np.array(rolling(sr_arr.tolist(), sw))
ax.plot(cumul, sr_roll, color="green", linewidth=2)
ax.set_ylabel("Success Rate")
ax.set_xlabel("Cumulative Environment Steps")
ax.set_ylim(-0.05, 1.05)
ax.set_title(f"Sample Efficiency\n(SR vs total env steps, {sw}-ep window)")
ax.grid(True, alpha=0.3)
# Mark where epsilon hits minimum
eps_min_ep = next((i for i, e in enumerate(
    [max(0.05, 1.0 * (0.997 ** ep)) for ep in range(EPISODES)]) if e <= 0.051), None)
if eps_min_ep is not None:
    ax.axvline(cumul[eps_min_ep], color="grey", linestyle="--", alpha=0.6,
               label=f"ε→min (ep {eps_min_ep})")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "entity_diagnostics.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print(f"Saved → {RESULTS_DIR}/entity_diagnostics.png")
print("Done.")
