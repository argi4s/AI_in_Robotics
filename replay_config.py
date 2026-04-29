# ─────────────────────────────────────────────────────────────────────────────
# replay_config.py  —  Replay & demo settings
#
# Edit this file to customise the visual demos without touching the scripts.
# ─────────────────────────────────────────────────────────────────────────────

# Time between agent moves (seconds).  Decrease for faster playback.
STEP_DELAY = 0.35

# Maze seeds — must match the seeds used during training.
# Changing these will load the same model on a different layout.
MAZE_SEED_NAV    = 42   # navigation DQN  (replay_dqn.py)
MAZE_SEED_ENTITY = 77   # entity DQN      (replay_entity.py)

# Grid size used for both replays (must match training).
GRID_SIZE = 12

# Maximum steps the agent is allowed per episode before the run ends.
MAX_STEPS = 300

# Pygame window — pixel size of each maze cell.
CELL_SIZE = 52
