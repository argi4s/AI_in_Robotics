import pickle
from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions
from models.sarsa import SARSAAgent
from utils.state_encoder import encode_state  # ✅ shared encoder


env = DungeonMazeEnv(render_mode="human", grid_size=10)
agent = SARSAAgent(action_size=len(Actions))

with open("sarsa_table.pkl", "rb") as f:
    agent.Q = pickle.load(f)

obs, _ = env.reset()
state = encode_state(obs)
done = False

while not done:
    action = agent.select_action(state, epsilon=0.0)  # ✅ pure greedy

    obs, _, terminated, truncated, _ = env.step(action)
    state = encode_state(obs)
    done = terminated or truncated