import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions
from envs.entity_dungeonworld_env import DungeonMazeEntityEnv, EntityActions
from models.dqn_network import DQN, DuelingDQN
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer
from utils.obs_utils import flatten_observation_v2, flatten_observation_entity

RESULTS_DIR  = "results"
SAVED_MODELS = "saved_models"
os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(SAVED_MODELS, exist_ok=True)

EVAL_WINDOW = 200


def summarise(successes, steps_list):
    sr    = float(np.mean(successes[-EVAL_WINDOW:]))
    valid = [s for s in steps_list[-EVAL_WINDOW:] if s is not None]
    avg   = float(np.mean(valid)) if valid else float("nan")
    return sr, avg


# ─── Config A: Vanilla DQN (no dueling) ──────────────────────────────────────

def run_A():
    TAG        = "vanilla_dqn"
    GRID_SIZE  = 12
    MAZE_SEED  = 42
    EPISODES   = 3000
    MAX_STEPS  = 300
    BATCH_SIZE = 128
    WARMUP     = 500
    TARGET_UPDATE_FREQ = 50
    TAU = 1.0

    env = DungeonMazeEnv(grid_size=GRID_SIZE, use_shaping=False)
    env.max_steps = MAX_STEPS

    obs, _    = env.reset(seed=MAZE_SEED)
    state     = flatten_observation_v2(obs, GRID_SIZE)
    state_dim  = state.shape[0]   # 10
    action_dim = len(Actions)     # 3

    # Vanilla DQN — no dueling streams
    q_net      = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)

    agent = DQNAgent(state_dim, action_dim, gamma=0.99, lr=1e-3,
                     epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.998,
                     use_double=True, tau=TAU)
    agent.set_networks(q_net, target_net)
    buffer = ReplayBuffer(capacity=100_000)

    successes, steps_list = [], []
    best_reward = -float("inf")
    device = agent.device

    print(f"\n{'='*60}")
    print(f"  Config A — Vanilla DQN (no dueling)  |  {GRID_SIZE}x{GRID_SIZE}, {EPISODES} eps")
    print(f"{'='*60}")

    for ep in range(EPISODES):
        obs, _  = env.reset(seed=MAZE_SEED)
        state   = flatten_observation_v2(obs, GRID_SIZE)
        total_r, success, steps = 0.0, False, 0

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = flatten_observation_v2(obs, GRID_SIZE)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
            steps   += 1
            if len(buffer) > WARMUP:
                agent.update(buffer.sample(BATCH_SIZE))
            if terminated:
                success = True
            if done:
                break

        agent.decay_epsilon()
        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if total_r > best_reward:
            best_reward = total_r
            torch.save(agent.q_net.state_dict(),
                       os.path.join(SAVED_MODELS, f"ablation_{TAG}_model.pth"))

        successes.append(1 if success else 0)
        steps_list.append(steps if success else None)

        if (ep + 1) % 100 == 0:
            sr, avg = summarise(successes, steps_list)
            avg_str = f"{avg:.1f}" if not np.isnan(avg) else "n/a"
            print(f"  ep {ep+1:4d}/{EPISODES} | SR(last 200): {sr:.1%} | "
                  f"avg steps: {avg_str} | eps: {agent.epsilon:.3f}")

    np.save(os.path.join(RESULTS_DIR, f"ablation_{TAG}_successes.npy"), np.array(successes))
    np.save(os.path.join(RESULTS_DIR, f"ablation_{TAG}_steps.npy"),
            np.array([s if s is not None else np.nan for s in steps_list]))

    sr, avg = summarise(successes, steps_list)
    print(f"\n  Config A done | SR: {sr:.1%} | Avg steps: "
          + (f"{avg:.1f}" if not np.isnan(avg) else "n/a"))
    return sr, avg


# ─── Shared entity training loop ─────────────────────────────────────────────

def run_entity(tag, label, disable_energy=False, disable_prepared_bonus=False):
    GRID_SIZE  = 12
    MAZE_SEED  = 77
    EPISODES   = 5000
    MAX_STEPS  = 300
    BATCH_SIZE = 128
    WARMUP     = 500
    TAU        = 0.005
    ENTITY_POSITIONS = {'tank': (6, 4), 'smart': (3, 4)}

    env = DungeonMazeEntityEnv(grid_size=GRID_SIZE, use_shaping=True,
                                entity_positions=ENTITY_POSITIONS,
                                use_perception=False,
                                disable_energy=disable_energy,
                                disable_prepared_bonus=disable_prepared_bonus)
    env.max_steps = MAX_STEPS

    obs, _    = env.reset(seed=MAZE_SEED)
    state     = flatten_observation_entity(obs, GRID_SIZE)
    state_dim  = state.shape[0]   # 17
    action_dim = len(EntityActions)  # 6

    q_net      = DuelingDQN(state_dim, action_dim)
    target_net = DuelingDQN(state_dim, action_dim)

    agent = DQNAgent(state_dim, action_dim, gamma=0.99, lr=1e-3,
                     epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.997,
                     use_double=True, tau=TAU)
    agent.set_networks(q_net, target_net)
    buffer = ReplayBuffer(capacity=100_000)

    successes, steps_list = [], []
    best_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"  {label}  |  {GRID_SIZE}x{GRID_SIZE}, {EPISODES} eps")
    print(f"  disable_energy={disable_energy}  disable_prepared_bonus={disable_prepared_bonus}")
    print(f"{'='*60}")

    for ep in range(EPISODES):
        obs, _  = env.reset(seed=MAZE_SEED)
        state   = flatten_observation_entity(obs, GRID_SIZE)
        total_r, success, steps = 0.0, False, 0

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = flatten_observation_entity(obs, GRID_SIZE)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
            steps   += 1
            if len(buffer) > WARMUP:
                agent.update(buffer.sample(BATCH_SIZE))
                agent.update_target()   # soft Polyak every grad step
            if terminated:
                success = True
            if done:
                break

        agent.decay_epsilon()
        if total_r > best_reward:
            best_reward = total_r
            torch.save(agent.q_net.state_dict(),
                       os.path.join(SAVED_MODELS, f"ablation_{tag}_model.pth"))

        successes.append(1 if success else 0)
        steps_list.append(steps if success else None)

        if (ep + 1) % 200 == 0:
            sr, avg = summarise(successes, steps_list)
            avg_str = f"{avg:.1f}" if not np.isnan(avg) else "n/a"
            print(f"  ep {ep+1:4d}/{EPISODES} | SR(last 200): {sr:.1%} | "
                  f"avg steps: {avg_str} | eps: {agent.epsilon:.3f}")

    np.save(os.path.join(RESULTS_DIR, f"ablation_{tag}_successes.npy"), np.array(successes))
    np.save(os.path.join(RESULTS_DIR, f"ablation_{tag}_steps.npy"),
            np.array([s if s is not None else np.nan for s in steps_list]))

    sr, avg = summarise(successes, steps_list)
    avg_str = f"{avg:.1f}" if not np.isnan(avg) else "n/a"
    print(f"\n  {label} done | SR: {sr:.1%} | Avg steps: {avg_str}")
    return sr, avg


def run_B():
    return run_entity("no_energy",  "Config B — Entity DQN (no energy)",
                      disable_energy=True, disable_prepared_bonus=False)


def run_C():
    return run_entity("no_bonus",   "Config C — Entity DQN (no +5 bonus)",
                      disable_energy=False, disable_prepared_bonus=True)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "C"], default=None,
                        help="Run a single config. Omit to run all three.")
    args = parser.parse_args()

    results = {}
    if args.config is None or args.config == "A":
        results["A"] = run_A()
    if args.config is None or args.config == "B":
        results["B"] = run_B()
    if args.config is None or args.config == "C":
        results["C"] = run_C()

    print("\n" + "=" * 50)
    print("  Ablation training complete")
    print("=" * 50)
    for cfg, (sr, avg) in results.items():
        avg_str = f"{avg:.1f}" if not np.isnan(avg) else "n/a"
        print(f"  Config {cfg}: SR={sr:.1%}  avg steps={avg_str}")
    print("\nRun  python -m train.ablation_table  to print the full table.")
