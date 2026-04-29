import numpy as np


def encode_state(obs):
    """
    Encodes an observation dict into a hashable state tuple for the Q-table.

    State = (x, y, direction, front_blocked)
      - x, y         : robot grid position
      - direction    : 0=N, 1=E, 2=S, 3=W
      - front_blocked: 1 if wall ahead (camera mean < 250), 0 if open
    """
    pos = obs["robot_position"]
    direction = obs["robot_direction"]
    camera = obs["robot_camera_view"]
    front_blocked = int(np.mean(camera) < 250)

    return (int(pos[0]), int(pos[1]), int(direction), front_blocked)