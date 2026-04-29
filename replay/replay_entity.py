"""
replay_entity.py — Pygame visualisation of the trained DQN+Entity agent.

Visual design
-------------
  Maze grid  : actual creature sprites (orc / wingedrat / halfling).
  Side panel : live stats, energy bar, last action, PERCEPTION section,
               entity status.
  Perception : shows what the robot currently sees —
               darkened sprite + CNN probs when entity is 2 blocks away,
               full-brightness sprite + SVM probs when entity is 1 block away.

Terminal narration
------------------
  Prints a thought-process log every time the robot scans or fights:

    [Step  12] CNN scan  (2 blocks, dark image, cost −5 energy)
               p_tank=82%  p_flying=12%  p_smart=6%
               → Belief: TANK  (82% confidence)
               → Preparing: FLEE

    [Step  15] SVM scan  (1 block, clear image)
               p_tank=97%  p_flying=2%  p_smart=1%
               → Confirmed: TANK  (97% confidence)

    [Step  15] COMBAT: FLEE  → WIN +25.0  ★ prepared bonus

Usage
-----
    python -m replay.replay_entity
"""

import os
import time
import torch
import numpy as np
import pygame

from envs.entity_dungeonworld_env import DungeonMazeEntityEnv, EntityActions, ENERGY_MAX
from utils.obs_utils import flatten_observation_entity
from models.dqn_network import DuelingDQN
from replay_config import STEP_DELAY, MAZE_SEED_ENTITY, GRID_SIZE, MAX_STEPS, CELL_SIZE

# ─── Config ────────────────────────────────────────────────────────────────────
MAZE_SEED  = MAZE_SEED_ENTITY
MODEL_PATH = os.path.join("saved_models", "best_dqn_entity_model.pth")
CELL       = CELL_SIZE
PANEL_W    = 320
MARGIN     = 4
SPRITE_SZ  = 56   # entity sprite size in the panel perception section

MAZE_W = GRID_SIZE * CELL
MAZE_H = GRID_SIZE * CELL
WIN_W  = MAZE_W + PANEL_W
WIN_H  = MAZE_H

# Which species image to use for each combat cluster
ENTITY_SPECIES_IMG = {
    'tank':   'orc',
    'flying': 'wingedrat',
    'smart':  'halfling',
}

CLUSTER_NAMES = ['tank', 'flying', 'smart']

# ─── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":          (20,  20,  30),
    "wall":        (50,  50,  60),
    "empty":       (230, 230, 235),
    "exit":        (255, 200,  50),
    "robot":       (30,  120, 255),
    "panel":       (30,  30,  45),
    "panel_line":  (60,  60,  80),
    "text":        (220, 220, 230),
    "text_dim":    (120, 120, 140),
    "win":         (80,  220,  80),
    "lose":        (220,  80,  80),
    "warn":        (255, 160,  40),
    "prepared":    (255, 215,   0),
    "tank":        (210,  55,  55),
    "flying":      ( 55,  90, 210),
    "smart":       ( 55, 180,  80),
    "tank_d":      ( 90,  45,  45),
    "flying_d":    ( 45,  55,  90),
    "smart_d":     ( 45,  80,  55),
    "act_move":    ( 80, 160, 255),
    "act_combat":  (255, 160,  40),
    "act_neutral": (160, 160, 180),
    "cnn_label":   (180, 120, 255),
    "svm_label":   ( 80, 200, 200),
}

ACTION_LABEL = {
    EntityActions.turn_right:    ("Turn Right",   "act_neutral"),
    EntityActions.turn_left:     ("Turn Left",    "act_neutral"),
    EntityActions.move_forwards: ("Move Forward", "act_move"),
    EntityActions.flee:          ("FLEE",         "act_combat"),
    EntityActions.use_bow:       ("USE BOW",      "act_combat"),
    EntityActions.use_sword:     ("USE SWORD",    "act_combat"),
}

ENTITY_META = {
    "tank":   {"letter": "T", "weak": "FLEE",  "col": "tank",   "col_d": "tank_d"},
    "flying": {"letter": "F", "weak": "BOW",   "col": "flying", "col_d": "flying_d"},
    "smart":  {"letter": "S", "weak": "SWORD", "col": "smart",  "col_d": "smart_d"},
}

DIR_VECTORS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


# ─── Sprite loading ────────────────────────────────────────────────────────────

def _load_sprite(species: str, size: int) -> pygame.Surface | None:
    """Load first image from images/<species>/, scale to size×size."""
    sdir = os.path.join("images", species)
    if not os.path.isdir(sdir):
        return None
    files = sorted(f for f in os.listdir(sdir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not files:
        return None
    try:
        surf = pygame.image.load(os.path.join(sdir, files[0])).convert()
        return pygame.transform.scale(surf, (size, size))
    except Exception:
        return None


def load_entity_sprites(cell_inner: int, panel_sz: int):
    """
    Returns two dicts: cell_sprites (for the grid) and panel_sprites (for panel).
    Each maps entity_type → pygame.Surface.
    """
    cell_s  = {}
    panel_s = {}
    for etype, species in ENTITY_SPECIES_IMG.items():
        cell_s[etype]  = _load_sprite(species, cell_inner)
        panel_s[etype] = _load_sprite(species, panel_sz)
    return cell_s, panel_s


def darken_surface(surf: pygame.Surface, gamma: int = 10) -> pygame.Surface:
    """Apply gamma darkening (simulating 2-block reduced lighting)."""
    try:
        arr = pygame.surfarray.array3d(surf).astype(np.float32) / 255.0
        arr = np.clip(arr ** gamma, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return pygame.surfarray.make_surface(arr)
    except Exception:
        result = surf.copy()
        dark = pygame.Surface(surf.get_size())
        dark.fill((0, 0, 0))
        dark.set_alpha(210)
        result.blit(dark, (0, 0))
        return result


def make_defeated_surface(surf: pygame.Surface) -> pygame.Surface:
    """Desaturate and darken a sprite to mark it as defeated."""
    result = surf.copy()
    overlay = pygame.Surface(surf.get_size())
    overlay.fill((0, 0, 0))
    overlay.set_alpha(155)
    result.blit(overlay, (0, 0))
    return result


# ─── Console narration ─────────────────────────────────────────────────────────

def _weapon_for(belief: str) -> str:
    return {'TANK': 'FLEE', 'FLYING': 'USE BOW', 'SMART': 'USE SWORD'}.get(belief, '?')


def _best_belief(probs):
    if probs is None:
        return '?', 0.0
    idx = int(np.argmax(probs))
    return CLUSTER_NAMES[idx].upper(), float(probs[idx])


def log_cnn_scan(step: int, etype: str, env):
    probs = env._scanned_2block.get(etype)
    belief, conf = _best_belief(probs)
    print(f"\n  ┌─ [Step {step:>3d}] CNN SCAN · 2 blocks ──────────────────────")
    print(f"  │  Entity detected ahead: {etype.upper()}")
    print(f"  │  Processing dark image...  (cost −5 energy)")
    if probs is not None:
        print(f"  │  p_tank={probs[0]*100:5.1f}%  "
              f"p_flying={probs[1]*100:5.1f}%  "
              f"p_smart={probs[2]*100:5.1f}%")
    print(f"  │  → Belief : {belief}  ({conf*100:.0f}% confidence)")
    print(f"  └─ → Prepare: {_weapon_for(belief)}")


def log_svm_scan(step: int, etype: str, env):
    probs = env._scanned_1block.get(etype)
    belief, conf = _best_belief(probs)
    print(f"  ┌─ [Step {step:>3d}] SVM SCAN · 1 block ───────────────────────")
    print(f"  │  Entity adjacent: {etype.upper()}")
    if probs is not None:
        print(f"  │  p_tank={probs[0]*100:5.1f}%  "
              f"p_flying={probs[1]*100:5.1f}%  "
              f"p_smart={probs[2]*100:5.1f}%")
    print(f"  └─ → Confirmed: {belief}  ({conf*100:.0f}% confidence)")


def log_combat(step: int, action, reward: float):
    action_name = str(action).split('.')[-1].upper().replace('_', ' ')
    is_win      = reward >= 20.0
    has_bonus   = reward >= 25.0
    result_str  = f"WIN  +{reward:.1f}" if is_win else f"MISS {reward:+.1f}"
    bonus_str   = "  ★ prepared bonus (CNN pre-scan paid off)" if has_bonus else ""
    print(f"  ──  [Step {step:>3d}] COMBAT: {action_name:<10} → {result_str}{bonus_str}")


def log_move(step: int, action):
    action_name = str(action).split('.')[-1].upper().replace('_', ' ')
    print(f"          [Step {step:>3d}] {action_name}")


# ─── Drawing helpers ───────────────────────────────────────────────────────────

def scan_entities(env):
    alive = {}
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            cell = env.maze.get_cell_item(x, y)
            if cell is not None and cell.type in ENTITY_META:
                alive[cell.type] = (x, y)
    return alive


def draw_text(surf, text, x, y, font, colour, anchor="topleft"):
    img = font.render(text, True, colour)
    r = img.get_rect(**{anchor: (x, y)})
    surf.blit(img, r)


def draw_robot_arrow(surf, cx, cy, direction, colour, size):
    dx, dy = DIR_VECTORS[direction]
    tip   = (cx + dx * size * 0.45, cy + dy * size * 0.45)
    left  = (cx + dy * size * 0.28 - dx * size * 0.28,
             cy - dx * size * 0.28 - dy * size * 0.28)
    right = (cx - dy * size * 0.28 - dx * size * 0.28,
             cy + dx * size * 0.28 - dy * size * 0.28)
    pygame.draw.polygon(surf, colour, [tip, left, right])


def draw_maze(surf, env, font_entity, alive, defeated_pos,
              cell_sprites, defeated_sprites):
    rx, ry = int(env.robot_position[0]), int(env.robot_position[1])
    tx, ty = int(env.target_position[0]), int(env.target_position[1])

    alive_cells    = {pos: etype for etype, pos in alive.items()}
    defeated_cells = {pos: etype for etype, pos in defeated_pos.items()}

    inner = CELL - 2 * MARGIN

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
                draw_text(surf, "EXIT", cx, cy, font_entity, C["bg"], "center")

            elif (gx, gy) in alive_cells:
                etype  = alive_cells[(gx, gy)]
                sprite = cell_sprites.get(etype)
                if sprite:
                    surf.blit(sprite, (px, py))
                else:
                    meta = ENTITY_META[etype]
                    pygame.draw.rect(surf, C[meta["col"]], rect, border_radius=6)
                    draw_text(surf, meta["letter"], cx, cy, font_entity,
                              (255, 255, 255), "center")

            elif (gx, gy) in defeated_cells:
                etype  = defeated_cells[(gx, gy)]
                sprite = defeated_sprites.get(etype)
                if sprite:
                    surf.blit(sprite, (px, py))
                else:
                    meta = ENTITY_META[etype]
                    pygame.draw.rect(surf, C[meta["col_d"]], rect, border_radius=6)
                    draw_text(surf, meta["letter"].lower(), cx, cy,
                              font_entity, C["text_dim"], "center")
            else:
                pygame.draw.rect(surf, C["empty"], rect, border_radius=3)

            if (gx, gy) == (rx, ry):
                draw_robot_arrow(surf, cx, cy, env.robot_direction, C["robot"], inner)


def draw_energy_bar(surf, x0, y, bar_w, energy, font_s):
    pct = max(0.0, energy / ENERGY_MAX)
    col = C["win"] if pct > 0.5 else C["warn"] if pct > 0.2 else C["lose"]
    draw_text(surf, f"Energy  {energy:>5.0f}/{ENERGY_MAX}", x0, y, font_s, col)
    y += 16
    pygame.draw.rect(surf, C["panel_line"], pygame.Rect(x0, y, bar_w, 8), border_radius=3)
    pygame.draw.rect(surf, col, pygame.Rect(x0, y, int(bar_w * pct), 8), border_radius=3)
    return y + 16


def draw_perception_section(surf, x0, y, env,
                             panel_sprites, dark_sprites,
                             font_b, font_s):
    """
    Show what the robot currently perceives.
    2-block → darkened sprite + CNN probabilities.
    1-block → full sprite + SVM probabilities.
    """
    entity_1 = env._entity_at_distance(1)
    entity_2 = env._entity_at_distance(2)

    draw_text(surf, "PERCEPTION", x0, y, font_s, C["text_dim"]); y += 18

    if entity_1 is not None:
        etype      = entity_1
        sprite     = panel_sprites.get(etype)
        probs      = env._scanned_1block.get(etype)
        scan_label = "SVM · 1 block  (clear)"
        label_col  = C["svm_label"]
    elif entity_2 is not None:
        etype      = entity_2
        sprite     = dark_sprites.get(etype)
        probs      = env._scanned_2block.get(etype)
        scan_label = "CNN · 2 blocks (dark)"
        label_col  = C["cnn_label"]
    else:
        draw_text(surf, "nothing in range", x0 + 4, y, font_s, C["text_dim"])
        return y + 18

    # Sprite on the left
    sprite_y = y
    if sprite:
        surf.blit(sprite, (x0, y))
    tx = x0 + SPRITE_SZ + 8   # text column starts after sprite

    draw_text(surf, scan_label, tx, y, font_s, label_col); y += 16

    if probs is not None:
        for cn, p in zip(CLUSTER_NAMES, probs):
            bw = max(2, int(p * 56))
            pygame.draw.rect(surf, C[cn],
                             pygame.Rect(tx, y + 3, bw, 7), border_radius=2)
            draw_text(surf, f"{p*100:3.0f}%", tx + 60, y, font_s, C[cn])
            y += 16

        best_i = int(np.argmax(probs))
        draw_text(surf,
                  f"→ {CLUSTER_NAMES[best_i].upper()} {probs[best_i]*100:.0f}%",
                  tx, y, font_b, label_col)
        y += 20
    else:
        draw_text(surf, "scanning...", tx, y, font_s, C["text_dim"]); y += 16

    return max(y, sprite_y + SPRITE_SZ + 4)


def draw_panel(surf, font_h, font_b, font_s,
               step, action, reward, total_reward,
               alive, defeated_pos, done, terminated,
               energy, env, prepared_flash,
               panel_sprites, dark_sprites):
    px = MAZE_W
    pygame.draw.rect(surf, C["panel"], pygame.Rect(px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surf, C["panel_line"], (px, 0), (px, WIN_H), 2)

    x0    = px + 14
    bar_w = PANEL_W - 28
    y     = 14

    # Title
    draw_text(surf, "DQN + ENTITIES", x0, y, font_h, C["text"])
    draw_text(surf, "auto replay  |  seed 77", x0 + 4, y + 20, font_s, C["text_dim"])
    y += 44
    pygame.draw.line(surf, C["panel_line"], (px + 6, y), (px + PANEL_W - 6, y), 1)
    y += 10

    # Stats
    draw_text(surf, f"Step   {step:>4d}", x0, y, font_b, C["text"]);         y += 22
    draw_text(surf, f"Reward {reward:>+6.1f}", x0, y, font_b, C["text"]);    y += 22
    draw_text(surf, f"Total  {total_reward:>7.1f}", x0, y, font_b, C["text"]); y += 22
    y = draw_energy_bar(surf, x0, y, bar_w, energy, font_s)
    y += 4

    # Last action
    pygame.draw.line(surf, C["panel_line"], (px + 6, y), (px + PANEL_W - 6, y), 1)
    y += 8
    draw_text(surf, "Last action", x0, y, font_s, C["text_dim"]); y += 18
    if action is not None:
        label, col_key = ACTION_LABEL[action]
        draw_text(surf, label, x0, y, font_h, C[col_key])
    if prepared_flash:
        draw_text(surf, "★ PREPARED +5", x0 + 104, y, font_s, C["prepared"])
    y += 28

    # Perception
    pygame.draw.line(surf, C["panel_line"], (px + 6, y), (px + PANEL_W - 6, y), 1)
    y += 8
    y = draw_perception_section(surf, x0, y, env,
                                panel_sprites, dark_sprites,
                                font_b, font_s)
    y += 6

    # Entities
    pygame.draw.line(surf, C["panel_line"], (px + 6, y), (px + PANEL_W - 6, y), 1)
    y += 8
    draw_text(surf, "Entities", x0, y, font_s, C["text_dim"]); y += 18

    for etype, meta in ENTITY_META.items():
        if etype in alive:
            pos    = alive[etype]
            col    = C[meta["col"]]
            status = f"{meta['letter']}  alive  [{pos[0]},{pos[1]}]"
        elif etype in defeated_pos:
            col    = C["text_dim"]
            status = f"{meta['letter'].lower()}  DEFEATED"
        else:
            col    = C["text_dim"]
            status = f"{meta['letter']}  not placed"
        pygame.draw.rect(surf, col, pygame.Rect(x0, y + 3, 9, 9), border_radius=2)
        draw_text(surf, status, x0 + 16, y, font_b, col);           y += 18
        draw_text(surf, f"   defeat: {meta['weak']}", x0, y, font_s, C["text_dim"]); y += 18

    # Legend (pinned to bottom)
    y = WIN_H - 60
    pygame.draw.line(surf, C["panel_line"], (px + 6, y), (px + PANEL_W - 6, y), 1)
    y += 8
    for line in ["Arrow=robot  orc/rat/half=entities",
                 "CNN=2-block  SVM=1-block  EXIT=goal"]:
        draw_text(surf, line, x0, y, font_s, C["text_dim"]); y += 16

    # Outcome banner
    if done:
        col = C["win"] if terminated else C["lose"]
        msg = "REACHED EXIT!" if terminated else "OUT OF ENERGY / TIME"
        pygame.draw.rect(surf, col,
                         pygame.Rect(px + 6, WIN_H - 80, PANEL_W - 12, 26),
                         border_radius=6)
        draw_text(surf, msg, px + PANEL_W // 2, WIN_H - 67,
                  font_h, C["bg"], "center")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("DQN Entity Replay — seed 77")
    clock = pygame.time.Clock()

    font_h      = pygame.font.SysFont("consolas", 16, bold=True)
    font_b      = pygame.font.SysFont("consolas", 14)
    font_s      = pygame.font.SysFont("consolas", 12)
    font_entity = pygame.font.SysFont("consolas", 20, bold=True)

    # Load sprites
    cell_inner = CELL - 2 * MARGIN
    cell_sprites, panel_sprites = load_entity_sprites(cell_inner, SPRITE_SZ)
    dark_sprites     = {k: darken_surface(v) for k, v in panel_sprites.items() if v}
    defeated_sprites = {k: make_defeated_surface(v) for k, v in cell_sprites.items() if v}

    env = DungeonMazeEntityEnv(
        grid_size=GRID_SIZE,
        use_shaping=True,
        entity_positions={'tank': (6, 4), 'smart': (3, 4)},
        use_perception=True,
    )
    env.max_steps = MAX_STEPS
    obs, _ = env.reset(seed=MAZE_SEED)
    state  = flatten_observation_entity(obs, GRID_SIZE)

    model = DuelingDQN(state.shape[0], len(EntityActions))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    alive         = scan_entities(env)
    defeated_pos  = {}
    total_reward  = 0.0
    step          = 0
    last_action   = None
    last_reward   = 0.0
    done          = False
    terminated    = False
    prepared_flash = False
    flash_timer   = 0

    # Console narration tracking
    logged_2block: set = set()
    logged_1block: set = set()

    print("\n" + "═" * 56)
    print("  DQN+Entity Replay  |  seed 77  |  12×12 maze")
    print("═" * 56)

    def render():
        screen.fill(C["bg"])
        draw_maze(screen, env, font_entity, alive, defeated_pos,
                  cell_sprites, defeated_sprites)
        draw_panel(screen, font_h, font_b, font_s,
                   step, last_action, last_reward, total_reward,
                   alive, defeated_pos, done, terminated,
                   env.energy, env, prepared_flash,
                   panel_sprites, dark_sprites)
        pygame.display.flip()

    render()
    time.sleep(STEP_DELAY * 2)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Snapshot scanned sets BEFORE step (to detect new scans after)
        scanned_2_before = set(env._scanned_2block.keys())
        scanned_1_before = set(env._scanned_1block.keys())

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_idx = model(state_t).argmax().item()

        last_action  = EntityActions(action_idx)
        alive_before = scan_entities(env)

        obs, reward, terminated, truncated, _ = env.step(action_idx)
        state = flatten_observation_entity(obs, GRID_SIZE)

        last_reward   = reward
        total_reward += reward
        step         += 1
        done          = terminated or truncated

        alive = scan_entities(env)
        for etype, pos in alive_before.items():
            if etype not in alive:
                defeated_pos[etype] = pos

        # ── Console narration ──────────────────────────────────────────────────
        new_2_scans = set(env._scanned_2block.keys()) - scanned_2_before
        new_1_scans = set(env._scanned_1block.keys()) - scanned_1_before

        for etype in new_2_scans:
            if etype not in logged_2block:
                logged_2block.add(etype)
                log_cnn_scan(step, etype, env)

        for etype in new_1_scans:
            if etype not in logged_1block:
                logged_1block.add(etype)
                log_svm_scan(step, etype, env)

        if action_idx in (3, 4, 5):
            log_combat(step, last_action, reward)
        else:
            log_move(step, last_action)

        # Prepared bonus flash (reward ≥ 25 on a combat action)
        prepared_flash = (action_idx in (3, 4, 5) and reward >= 25.0)
        if prepared_flash:
            flash_timer = 8
        elif flash_timer > 0:
            flash_timer -= 1
            prepared_flash = flash_timer > 0

        render()
        clock.tick(1000)
        time.sleep(STEP_DELAY)

    # Final console summary
    outcome = "REACHED EXIT" if terminated else "OUT OF ENERGY / TIME"
    print(f"\n{'═' * 56}")
    print(f"  Episode end: {outcome}")
    print(f"  Steps: {step}  |  Total reward: {total_reward:.1f}")
    print(f"{'═' * 56}\n")

    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    main()
