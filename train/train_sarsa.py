from envs.simple_dungeonworld_env import DungeonMazeEnv
from models.sarsa import SARSAAgent
from utils.state_encoder import encode_state  # ✅ shared encoder
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


env = DungeonMazeEnv(grid_size=10)
agent = SARSAAgent(action_size=env.action_space.n)

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01

episode_rewards = []
best_reward = -float("inf")
best_Q = None

for episode in range(1000):
    obs, _ = env.reset()
    state = encode_state(obs)
    action = agent.select_action(state, epsilon)
    total_reward = 0

    while True:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = encode_state(next_obs)
        next_action = agent.select_action(next_state, epsilon)

        agent.update(state, action, reward, next_state, next_action, alpha, gamma)

        state = next_state
        action = next_action
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if total_reward > best_reward:
        best_reward = total_reward
        best_Q = copy.deepcopy(agent.Q)

    episode_rewards.append(total_reward)
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Best: {best_reward:.2f}, Epsilon: {epsilon:.3f}")

# ===== SAVE BEST MODEL =====
with open("sarsa_table.pkl", "wb") as f:
    pickle.dump(best_Q, f)

# ===== PLOT =====
plt.plot(episode_rewards)
plt.title("SARSA Training")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig("sarsa_training.png")
plt.show()