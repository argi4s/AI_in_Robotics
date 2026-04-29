"""
replay_dqn.py — Polished Pygame replay of the trained DQN navigation agent.

Side panel shows:
  - Live step / reward / total reward
  - Last action with colour coding
  - Robot position, heading (compass), distance to goal
  - Wall sensors (front / left / right)
  - Outcome banner on completion

Usage
-----
    python -m replay.replay_dqn
"""

import time
import torch
import numpy as np
import pygame

from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions
from utils.obs_utils import flatten_observation, flatten_observation_v2
from models.dqn_network import DQN, DuelingDQN
from replay_config import STEP_DELAY, MAZE_SEED_NAV, GRID_SIZE, MAX_STEPS, CELL_SIZE

# ─── Config ────────────────────────────────────────────────────────────────────
MAZE_SEED  = MAZE_SEED_NAV
MODEL_PATH = "saved_models/ablation_vanilla_dqn_model.pth"

CELL    = CELL_SIZE
MARGIN  = 4
PANEL_W = 280

MAZE_W = GRID_SIZE * CELL
MAZE_H = GRID_SIZE * CELL
WIN_W  = MAZE_W + PANEL_W
WIN_H  = MAZE_H

# ─── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":         (20,  20,  30),
    "wall":       (50,  50,  60),
    "empty":      (230, 230, 235),
    "exit":       (255, 200,  50),
    "robot":      (30,  120, 255),
    "panel":      (30,  30,  45),
    "divider":    (60,  60,  80),
    "text":       (220, 220, 230),
    "dim":        (120, 120, 140),
    "win":        (80,  220,  80),
    "lose":       (220,  80,  80),
    "warn":       (255, 160,  40),
    "move":       (80,  180, 255),
    "turn":       (180, 180,  80),
    "sensor_on":  (255, 100,  80),
    "sensor_off": (60,   70,  90),
    "bar_bg":     (50,  50,  65),
}

COMPASS = {0: "N", 1: "E", 2: "S", 3: "W"}
DIR_VEC = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

ACTION_META = {
    Actions.turn_right:    ("Turn Right  ▶",  "turn"),
    Actions.turn_left:     ("◀ Turn Left",    "turn"),
    Actions.move_forwards: ("Move Forward ↑", "move"),
}


# ─── Drawing helpers ───────────────────────────────────────────────────────────

def txt(surf, text, x, y, font, colour, anchor="topleft"):
    img = font.render(text, True, colour)
    r = img.get_rect(**{anchor: (x, y)})
    surf.blit(img, r)


def divider(surf, y, px):
    pygame.draw.line(surf, C["divider"], (px + 6, y), (px + PANEL_W - 6, y), 1)


def draw_robot_arrow(surf, cx, cy, direction, size):
    dx, dy = DIR_VEC[direction]
    tip   = (cx + dx * size * 0.45, cy + dy * size * 0.45)
    left  = (cx + dy * size * 0.28 - dx * size * 0.28,
             cy - dx * size * 0.28 - dy * size * 0.28)
    right = (cx - dy * size * 0.28 - dx * size * 0.28,
             cy + dx * size * 0.28 - dy * size * 0.28)
    pygame.draw.polygon(surf, C["robot"], [tip, left, right])


def draw_maze(surf, env, font_exit):
    rx, ry = int(env.robot_position[0]), int(env.robot_position[1])
    tx, ty = int(env.target_position[0]), int(env.target_position[1])
    inner  = CELL - 2 * MARGIN

    for gx in range(GRID_SIZE):
        for gy in range(GRID_SIZE):
            px   = gx * CELL + MARGIN
            py   = gy * CELL + MARGIN
            rect = pygame.Rect(px, py, inner, inner)
            cell = env.maze.get_cell_item(gx, gy)
            cx   = px + inner // 2
            cy   = py + inner // 2

            if cell is not None and cell.type == "wall":
                pygame.draw.rect(surf, C["wall"], rect, border_radius=3)
            elif (gx, gy) == (tx, ty):
                pygame.draw.rect(surf, C["exit"], rect, border_radius=6)
                txt(surf, "EXIT", cx, cy, font_exit, C["bg"], "center")
            else:
                pygame.draw.rect(surf, C["empty"], rect, border_radius=3)

            if (gx, gy) == (rx, ry):
                draw_robot_arrow(surf, cx, cy, env.robot_direction, inner)


def draw_sensor_pill(surf, x, y, label, blocked, font):
    col = C["sensor_on"] if blocked else C["sensor_off"]
    pygame.draw.rect(surf, col, pygame.Rect(x, y, 52, 18), border_radius=4)
    txt(surf, label, x + 26, y + 9, font, C["text"], "center")


def draw_progress_bar(surf, x, y, w, h, pct, col):
    pygame.draw.rect(surf, C["bar_bg"], pygame.Rect(x, y, w, h), border_radius=3)
    pygame.draw.rect(surf, col,         pygame.Rect(x, y, max(4, int(w * pct)), h),
                     border_radius=3)


def draw_panel(surf, fonts, env, step, last_action, last_reward,
               total_reward, done, terminated, max_steps):
    fh, fb, fs = fonts
    px = MAZE_W
    x0 = px + 14
    bw = PANEL_W - 28   # bar / content width
    y  = 14

    pygame.draw.rect(surf, C["panel"], pygame.Rect(px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surf, C["divider"], (px, 0), (px, WIN_H), 2)

    # ── Title ──────────────────────────────────────────────────────────────────
    txt(surf, "DQN NAVIGATION", x0, y, fh, C["text"]); y += 20
    txt(surf, f"12×12 maze  ·  seed {MAZE_SEED}", x0, y, fs, C["dim"]); y += 14
    txt(surf, "Double DQN + Dueling architecture", x0, y, fs, C["dim"]); y += 22
    divider(surf, y, px); y += 10

    # ── Stats ──────────────────────────────────────────────────────────────────
    txt(surf, f"Step", x0,      y, fs, C["dim"])
    txt(surf, f"{step:>4d} / {max_steps}", x0 + 40, y, fb, C["text"]); y += 4
    step_pct = min(1.0, step / max_steps)
    step_col = C["win"] if step_pct < 0.6 else C["warn"] if step_pct < 0.85 else C["lose"]
    draw_progress_bar(surf, x0, y + 14, bw, 5, step_pct, step_col); y += 26

    txt(surf, f"Reward      {last_reward:>+6.1f}", x0, y, fb, C["text"]); y += 20
    txt(surf, f"Total       {total_reward:>+7.1f}", x0, y, fb, C["text"]); y += 22
    divider(surf, y, px); y += 10

    # ── Last action ────────────────────────────────────────────────────────────
    txt(surf, "Last Action", x0, y, fs, C["dim"]); y += 18
    if last_action is not None:
        label, col_key = ACTION_META[last_action]
        txt(surf, label, x0, y, fh, C[col_key])
    y += 28
    divider(surf, y, px); y += 10

    # ── Robot state ────────────────────────────────────────────────────────────
    txt(surf, "Robot", x0, y, fs, C["dim"]); y += 18
    rx, ry = env.robot_position
    tx, ty = env.target_position
    heading = COMPASS[env.robot_direction]

    txt(surf, f"Pos    ({int(rx):>2d}, {int(ry):>2d})", x0, y, fb, C["text"]); y += 20
    txt(surf, f"Facing  {heading}", x0, y, fb, C["text"]); y += 20

    dist = float(np.sqrt((tx - rx)**2 + (ty - ry)**2))
    max_dist = float(GRID_SIZE * 1.414)
    dist_col = C["win"] if dist < 3 else C["warn"] if dist < 6 else C["text"]
    txt(surf, f"Target ({int(tx):>2d}, {int(ty):>2d})  d={dist:.1f}", x0, y, fb, dist_col)
    y += 6
    draw_progress_bar(surf, x0, y + 14, bw, 5,
                      max(0.0, 1.0 - dist / max_dist), dist_col); y += 28
    divider(surf, y, px); y += 10

    # ── Wall sensors ───────────────────────────────────────────────────────────
    txt(surf, "Wall Sensors", x0, y, fs, C["dim"]); y += 18

    obs = env.get_observations()
    front_blocked = float(np.mean(obs["robot_camera_view"]) < 250)
    left_blocked  = obs.get("left_blocked",  0)
    right_blocked = obs.get("right_blocked", 0)

    draw_sensor_pill(surf, x0,        y, "FRONT", front_blocked, fs)
    draw_sensor_pill(surf, x0 + 60,   y, "LEFT",  left_blocked,  fs)
    draw_sensor_pill(surf, x0 + 120,  y, "RIGHT", right_blocked, fs)
    y += 30
    divider(surf, y, px); y += 10

    # ── Legend ─────────────────────────────────────────────────────────────────
    txt(surf, "Arrow = robot direction", x0, y, fs, C["dim"]); y += 16
    txt(surf, "Yellow = EXIT cell",      x0, y, fs, C["dim"]); y += 16
    txt(surf, "Red sensor = wall ahead", x0, y, fs, C["dim"]); y += 16

    # ── Outcome banner ─────────────────────────────────────────────────────────
    if done:
        col = C["win"] if terminated else C["lose"]
        msg = "REACHED EXIT!" if terminated else "OUT OF STEPS"
        pygame.draw.rect(surf, col,
                         pygame.Rect(px + 6, WIN_H - 42, PANEL_W - 12, 28),
                         border_radius=6)
        txt(surf, msg, px + PANEL_W // 2, WIN_H - 28, fh, C["bg"], "center")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"DQN Navigation Replay  ·  {GRID_SIZE}×{GRID_SIZE}  ·  seed {MAZE_SEED}")
    clock  = pygame.time.Clock()

    fh = pygame.font.SysFont("consolas", 15, bold=True)
    fb = pygame.font.SysFont("consolas", 13)
    fs = pygame.font.SysFont("consolas", 11)
    fe = pygame.font.SysFont("consolas", 18, bold=True)

    env = DungeonMazeEnv(render_mode=None, grid_size=GRID_SIZE, use_shaping=False)
    env.max_steps = MAX_STEPS
    obs, _ = env.reset(seed=MAZE_SEED)

    # Auto-detect architecture and input size from the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    is_dueling = "fc_value.weight" in checkpoint
    input_dim  = checkpoint["fc1.weight"].shape[1]
    obs_fn     = flatten_observation_v2 if input_dim == 10 else flatten_observation
    NetClass   = DuelingDQN if is_dueling else DQN
    print(f"  Model: {'DuelingDQN' if is_dueling else 'Vanilla DQN'}  "
          f"input_dim={input_dim}  ({'v2 10-D' if input_dim == 10 else '6-D'})")

    state = obs_fn(obs, GRID_SIZE) if input_dim == 10 else obs_fn(obs)
    model = NetClass(input_dim, len(Actions))
    model.load_state_dict(checkpoint)
    model.eval()

    step         = 0
    total_reward = 0.0
    last_action  = None
    last_reward  = 0.0
    done         = False
    terminated   = False

    print(f"\n{'═'*50}")
    print(f"  DQN Navigation Replay  ·  {GRID_SIZE}×{GRID_SIZE}  ·  seed {MAZE_SEED}")
    print(f"{'═'*50}")

    def render():
        screen.fill(C["bg"])
        draw_maze(screen, env, fe)
        draw_panel(screen, (fh, fb, fs), env, step, last_action,
                   last_reward, total_reward, done, terminated, env.max_steps)
        pygame.display.flip()

    render()
    time.sleep(STEP_DELAY * 2)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return

        with torch.no_grad():
            action_idx = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

        last_action = Actions(action_idx)
        obs, reward, terminated, truncated, _ = env.step(action_idx)
        state = obs_fn(obs, GRID_SIZE) if input_dim == 10 else obs_fn(obs)
        last_reward  = reward
        total_reward += reward
        step        += 1
        done         = terminated or truncated

        action_name = str(last_action).split(".")[-1].upper().replace("_", " ")
        print(f"  [{step:>3d}] {action_name:<16}  r={reward:+.2f}  total={total_reward:+.1f}")

        render()
        clock.tick(1000)
        time.sleep(STEP_DELAY)

    outcome = "REACHED EXIT" if terminated else "OUT OF STEPS"
    print(f"\n{'═'*50}")
    print(f"  {outcome}  ·  steps={step}  ·  total reward={total_reward:.1f}")
    print(f"{'═'*50}\n")

    render()
    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    main()
