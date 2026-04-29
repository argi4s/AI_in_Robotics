"""
manual_entity.py — Human-controlled DungeonMaze with combat entities.

Controls
--------
  W         move forward
  A         turn left
  D         turn right
  1         FLEE    (beats Tank  T)
  2         USE BOW (beats Flying F)
  3         USE SWORD (beats Smart S)
  R         reset maze (new random seed)
  Q / Esc   quit

Visual style matches the DQN replay:
  - Orc / wingedrat / halfling sprites in the grid
  - Perception panel shows darkened sprite at 2-block range,
    full sprite at 1-block range (mirrors what the DQN agent perceives)

Usage
-----
    python -m manual.manual_entity
"""

import os
import random
import numpy as np
import pygame

from envs.entity_dungeonworld_env import DungeonMazeEntityEnv, EntityActions, ENERGY_MAX

# ─── Config ────────────────────────────────────────────────────────────────────
GRID_SIZE  = 12
CELL       = 52
PANEL_W    = 320
MARGIN     = 4
SPRITE_SZ  = 56   # sprite size in the panel perception section

MAZE_W = GRID_SIZE * CELL
MAZE_H = GRID_SIZE * CELL
WIN_W  = MAZE_W + PANEL_W
WIN_H  = MAZE_H

ENTITY_SPECIES_IMG = {
    'tank':   'orc',
    'flying': 'wingedrat',
    'smart':  'halfling',
}

# ─── Key → action mapping ──────────────────────────────────────────────────────
KEY_MAP = {
    pygame.K_w:   EntityActions.move_forwards,
    pygame.K_a:   EntityActions.turn_left,
    pygame.K_d:   EntityActions.turn_right,
    pygame.K_1:   EntityActions.flee,
    pygame.K_2:   EntityActions.use_bow,
    pygame.K_3:   EntityActions.use_sword,
    pygame.K_KP1: EntityActions.flee,
    pygame.K_KP2: EntityActions.use_bow,
    pygame.K_KP3: EntityActions.use_sword,
}

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
    "tank":   {"letter": "T", "weak": "FLEE  [1]",  "col": "tank",   "col_d": "tank_d"},
    "flying": {"letter": "F", "weak": "BOW   [2]",  "col": "flying", "col_d": "flying_d"},
    "smart":  {"letter": "S", "weak": "SWORD [3]",  "col": "smart",  "col_d": "smart_d"},
}

WEAPON_FOR = {'tank': 'FLEE  [1]', 'flying': 'BOW   [2]', 'smart': 'SWORD [3]'}

DIR_VECTORS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


# ─── Sprite loading ────────────────────────────────────────────────────────────

def _load_sprite(species: str, size: int):
    sdir  = os.path.join("images", species)
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
    cell_s  = {}
    panel_s = {}
    for etype, species in ENTITY_SPECIES_IMG.items():
        cell_s[etype]  = _load_sprite(species, cell_inner)
        panel_s[etype] = _load_sprite(species, panel_sz)
    return cell_s, panel_s


def darken_surface(surf: pygame.Surface, gamma: int = 10) -> pygame.Surface:
    try:
        arr = pygame.surfarray.array3d(surf).astype(np.float32) / 255.0
        arr = np.clip(arr ** gamma, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return pygame.surfarray.make_surface(arr)
    except Exception:
        result = surf.copy()
        dark   = pygame.Surface(surf.get_size())
        dark.fill((0, 0, 0))
        dark.set_alpha(210)
        result.blit(dark, (0, 0))
        return result


def make_defeated_surface(surf: pygame.Surface) -> pygame.Surface:
    result  = surf.copy()
    overlay = pygame.Surface(surf.get_size())
    overlay.fill((0, 0, 0))
    overlay.set_alpha(155)
    result.blit(overlay, (0, 0))
    return result


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
    r   = img.get_rect(**{anchor: (x, y)})
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
    inner          = CELL - 2 * MARGIN

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
    Shows what the robot currently sees ahead — mirrors the DQN replay panel.
    Oracle mode (no classifier): visual only, no probability bars.
      2-block range → darkened sprite  (simulates CNN input)
      1-block range → full sprite      (simulates SVM input)
    """
    entity_1 = env._entity_at_distance(1)
    entity_2 = env._entity_at_distance(2)

    draw_text(surf, "PERCEPTION", x0, y, font_s, C["text_dim"]); y += 18

    if entity_1 is not None:
        etype      = entity_1
        sprite     = panel_sprites.get(etype)
        scan_label = "SVM range · 1 block"
        label_col  = C["svm_label"]
    elif entity_2 is not None:
        etype      = entity_2
        sprite     = dark_sprites.get(etype)
        scan_label = "CNN range · 2 blocks"
        label_col  = C["cnn_label"]
    else:
        draw_text(surf, "nothing in range", x0 + 4, y, font_s, C["text_dim"])
        return y + 18

    sprite_y = y
    if sprite:
        surf.blit(sprite, (x0, y))
    tx = x0 + SPRITE_SZ + 8

    draw_text(surf, scan_label, tx, y, font_s, label_col);  y += 18
    draw_text(surf, etype.upper(), tx, y, font_b, C[etype]); y += 20
    draw_text(surf, f"→ use: {WEAPON_FOR[etype]}", tx, y, font_s, C["act_combat"])
    y += 18

    return max(y, sprite_y + SPRITE_SZ + 4)


def draw_panel(surf, font_h, font_b, font_s,
               step, action, reward, total_reward,
               alive, defeated_pos, done, terminated,
               bad_action_flash, energy, env, seed,
               panel_sprites, dark_sprites):
    px = MAZE_W
    pygame.draw.rect(surf, C["panel"], pygame.Rect(px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surf, C["panel_line"], (px, 0), (px, WIN_H), 2)

    x0    = px + 14
    bar_w = PANEL_W - 28
    y     = 14

    # Title
    draw_text(surf, "DUNGEON MAZE", x0, y, font_h, C["text"])
    draw_text(surf, f"manual control  |  seed {seed}", x0 + 4, y + 20, font_s, C["text_dim"])
    y += 44
    pygame.draw.line(surf, C["panel_line"], (px + 6, y), (px + PANEL_W - 6, y), 1)
    y += 10

    # Stats
    draw_text(surf, f"Step   {step:>4d}", x0, y, font_b, C["text"]);          y += 22
    draw_text(surf, f"Reward {reward:>+6.1f}", x0, y, font_b, C["text"]);     y += 22
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
    if bad_action_flash:
        draw_text(surf, "✗ WRONG  −3 energy", x0 + 104, y, font_s, C["lose"])
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
        draw_text(surf, status, x0 + 16, y, font_b, col);              y += 18
        draw_text(surf, f"   defeat: {meta['weak']}", x0, y, font_s, C["text_dim"]); y += 18

    # Controls (pinned near bottom)
    y = WIN_H - 148
    pygame.draw.line(surf, C["panel_line"], (px + 6, y), (px + PANEL_W - 6, y), 1)
    y += 8
    draw_text(surf, "Controls", x0, y, font_s, C["text_dim"]); y += 18
    for key, desc, col_key in [
        ("W",   "Move forward",   "act_move"),
        ("A/D", "Turn",           "act_neutral"),
        ("1",   "FLEE   (Tank)",  "act_combat"),
        ("2",   "BOW    (Flying)","act_combat"),
        ("3",   "SWORD  (Smart)", "act_combat"),
        ("R",   "Reset maze",     "text_dim"),
        ("Q",   "Quit",           "text_dim"),
    ]:
        draw_text(surf, f"[{key}]", x0,      y, font_s, C[col_key])
        draw_text(surf, desc,       x0 + 44, y, font_s, C["text"])
        y += 17

    # Outcome banner
    if done:
        col = C["win"] if terminated else C["lose"]
        msg = "REACHED EXIT!  [R] reset" if terminated else "OUT OF ENERGY  [R] reset"
        pygame.draw.rect(surf, col,
                         pygame.Rect(px + 6, WIN_H - 34, PANEL_W - 12, 26),
                         border_radius=6)
        draw_text(surf, msg, px + PANEL_W // 2, WIN_H - 21,
                  font_h, C["bg"], "center")


# ─── Main ──────────────────────────────────────────────────────────────────────

def reset_game(seed=None):
    env = DungeonMazeEntityEnv(
        grid_size=GRID_SIZE,
        use_shaping=False,
        entity_positions={'tank': (6, 4), 'smart': (3, 4)},
    )
    env.max_steps = 300
    seed = seed if seed is not None else random.randint(0, 9999)
    env.reset(seed=seed)
    alive        = scan_entities(env)
    defeated_pos = {}
    return env, alive, defeated_pos, seed


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    font_h      = pygame.font.SysFont("consolas", 16, bold=True)
    font_b      = pygame.font.SysFont("consolas", 14)
    font_s      = pygame.font.SysFont("consolas", 12)
    font_entity = pygame.font.SysFont("consolas", 20, bold=True)

    cell_inner = CELL - 2 * MARGIN
    cell_sprites, panel_sprites = load_entity_sprites(cell_inner, SPRITE_SZ)
    dark_sprites     = {k: darken_surface(v) for k, v in panel_sprites.items() if v}
    defeated_sprites = {k: make_defeated_surface(v) for k, v in cell_sprites.items() if v}

    env, alive, defeated_pos, seed = reset_game(seed=77)
    pygame.display.set_caption(f"Dungeon Maze — Manual Control  (seed {seed})")

    total_reward     = 0.0
    step             = 0
    last_action      = None
    last_reward      = 0.0
    done             = False
    terminated       = False
    bad_action_flash = False
    flash_timer      = 0

    def render():
        screen.fill(C["bg"])
        draw_maze(screen, env, font_entity, alive, defeated_pos,
                  cell_sprites, defeated_sprites)
        draw_panel(screen, font_h, font_b, font_s,
                   step, last_action, last_reward, total_reward,
                   alive, defeated_pos, done, terminated,
                   bad_action_flash, env.energy, env, seed,
                   panel_sprites, dark_sprites)
        pygame.display.flip()

    render()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    return

                if event.key == pygame.K_r:
                    env, alive, defeated_pos, seed = reset_game()
                    pygame.display.set_caption(
                        f"Dungeon Maze — Manual Control  (seed {seed})")
                    total_reward = last_reward = step = 0.0
                    last_action  = None
                    done         = terminated = bad_action_flash = False
                    flash_timer  = 0
                    render()
                    continue

                if done:
                    continue

                if event.key in KEY_MAP:
                    action_idx   = int(KEY_MAP[event.key])
                    alive_before = scan_entities(env)

                    _, reward, terminated, truncated, _ = env.step(action_idx)

                    last_action   = EntityActions(action_idx)
                    last_reward   = reward
                    total_reward += reward
                    step         += 1
                    done          = terminated or truncated

                    alive = scan_entities(env)
                    for etype, pos in alive_before.items():
                        if etype not in alive:
                            defeated_pos[etype] = pos

                    bad_action_flash = (reward <= -3.0 and action_idx in (3, 4, 5))
                    flash_timer      = 90
                    render()

        if bad_action_flash:
            flash_timer -= 1
            if flash_timer <= 0:
                bad_action_flash = False
                render()

        clock.tick(60)


if __name__ == "__main__":
    main()
