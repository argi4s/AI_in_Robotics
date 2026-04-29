import numpy as np


def flatten_observation(obs):
    """
    Encodes the observation dict into a flat float32 vector for the DQN.

    State vector (6 dims):
        [x, y, direction, front_blocked, target_x, target_y]

    front_blocked: 1.0 if wall ahead (camera mean < 250), 0.0 if open.
    This matches the signal used in the tabular SARSA/Q-learning encoder,
    keeping all three methods comparable.

    Used by the original 10×10 training script. Kept intact for backwards
    compatibility. For larger mazes use flatten_observation_v2.
    """
    front_blocked = float(np.mean(obs["robot_camera_view"]) < 250)

    return np.concatenate([
        obs["robot_position"],          # 2
        [obs["robot_direction"]],       # 1
        [front_blocked],                # 1
        obs["target_position"],         # 2
    ]).astype(np.float32)              # total: 6


def flatten_observation_v2(obs, grid_size: int) -> np.ndarray:
    """
    Enhanced 10-D normalised state vector for larger mazes.

    Dimensions
    ----------
    0: x  / (grid_size - 1)         robot column, normalised to [0, 1]
    1: y  / (grid_size - 1)         robot row,    normalised to [0, 1]
    2: direction / 3                heading (0=N,1=E,2=S,3=W), normalised to [0, 1]
    3: front_blocked                binary wall sensor — 1 if wall ahead, 0 if open
    4: tx / (grid_size - 1)         target column, normalised to [0, 1]
    5: ty / (grid_size - 1)         target row,    normalised to [0, 1]
    6: (tx - x) / (grid_size - 1)  relative x offset to target, in (−1, 1)
    7: (ty - y) / (grid_size - 1)  relative y offset to target, in (−1, 1)
    8: left_blocked                 binary wall sensor — 1 if wall to the left, 0 if open
    9: right_blocked                binary wall sensor — 1 if wall to the right, 0 if open

    Why normalise?
    --------------
    In the original 6-D state the raw coordinates for a 16×16 grid range up
    to 15. Inputs of different magnitudes cause gradients to be dominated by
    the largest-valued features. Dividing everything by (grid_size − 1) maps
    all features to roughly [0, 1] and removes this imbalance.

    Why add the relative offset (dims 6-7)?
    ----------------------------------------
    Even though absolute positions (dims 0-1, 4-5) encode the same information,
    the network must implicitly learn to subtract them to compute "how far am I
    from the goal?" Adding the difference explicitly provides a direct gradient
    signal for moving toward the exit from the very first episode, rather than
    waiting for the network to discover the subtraction operation on its own.

    Why add left/right sensors (dims 8-9)?
    ----------------------------------------
    At each step the agent knows front/left/right wall status — the same
    perceptual information a human would use to navigate a maze. This makes
    turn decisions locally grounded: turning into a wall is always wrong, and
    the network can learn this from the very first crash. Without these sensors
    the agent must infer open passages indirectly from position changes across
    consecutive steps, which is much harder to generalise across random mazes.

    Args
    ----
    obs       : dict observation from DungeonMazeEnv
    grid_size : the maze grid size passed from the training script

    Returns
    -------
    numpy float32 array of shape (10,)
    """
    gs = float(grid_size - 1)   # normalisation denominator

    x,  y  = float(obs["robot_position"][0]),  float(obs["robot_position"][1])
    tx, ty = float(obs["target_position"][0]), float(obs["target_position"][1])
    front_blocked = float(np.mean(obs["robot_camera_view"]) < 250)

    return np.array([
        x  / gs,                       # 0: robot x, normalised
        y  / gs,                       # 1: robot y, normalised
        obs["robot_direction"] / 3.0,  # 2: heading, normalised
        front_blocked,                 # 3: front wall sensor
        tx / gs,                       # 4: target x, normalised
        ty / gs,                       # 5: target y, normalised
        (tx - x) / gs,                 # 6: relative x to target
        (ty - y) / gs,                 # 7: relative y to target
        float(obs["left_blocked"]),    # 8: left wall sensor
        float(obs["right_blocked"]),   # 9: right wall sensor
    ], dtype=np.float32)


def flatten_observation_entity(obs, grid_size: int) -> np.ndarray:
    """
    17-D state vector for the entity environment.

    Dims  0-9 : base navigation state (flatten_observation_v2)
    Dims 10-12: entity at 1 block  — tank / flying / smart
    Dims 13-15: entity at 2 blocks — tank / flying / smart
                (0 if line-of-sight is blocked by a wall or entity)
    Dim  16   : energy normalised  — energy / ENERGY_MAX ∈ [0, 1]

    The 2-block sensors give the agent advance warning so it can commit
    to the correct weapon before the entity is adjacent (prepared-weapon
    bonus of +5 if the agent acts correctly having seen it at 2 blocks).

    The energy dimension teaches the agent to be efficient: wasted combat
    actions and unnecessary steps deplete energy, reducing the exit bonus.

    Returns
    -------
    numpy float32 array of shape (17,)
    """
    base = flatten_observation_v2(obs, grid_size)
    sensors = np.array([
        float(obs["tank_front"]),
        float(obs["flying_front"]),
        float(obs["smart_front"]),
        float(obs["tank_2block"]),
        float(obs["flying_2block"]),
        float(obs["smart_2block"]),
        float(obs["energy"][0]),
    ], dtype=np.float32)
    return np.concatenate([base, sensors])