import pickle
from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions
from models.q_learning import QLearningAgent


def encode_state(obs):
    pos = obs["robot_position"]
    direction = obs["robot_direction"]
    return (int(pos[0]), int(pos[1]), int(direction))


env = DungeonMazeEnv(render_mode="human", grid_size=8)

agent = QLearningAgent(action_size=len(Actions))

# ✅ LOAD trained Q-table
with open("q_table.pkl", "rb") as f:
    agent.Q = pickle.load(f)

obs, _ = env.reset()
state = encode_state(obs)

done = False

while not done:
    # ✅ PURE exploitation
    action_idx = agent.select_action(state, epsilon=0.0)
    action = list(Actions)[action_idx]

    obs, _, terminated, truncated, _ = env.step(action)
    state = encode_state(obs)

    done = terminated or truncated