"""
entity_dungeonworld_env.py — DungeonMaze with behaviour-cluster entities,
energy management, and 2-block advance detection.

Actions (6 total)
-----------------
  0: turn_right
  1: turn_left
  2: move_forwards
  3: flee       — defeats Tank,   wrong vs Flying / Smart
  4: use_bow    — defeats Flying, wrong vs Tank   / Smart
  5: use_sword  — defeats Smart,  wrong vs Tank   / Flying

Entities
--------
  TankEntity   — defeated by flee   (+20).  Wrong action: −3, −10 energy.
  FlyingEntity — defeated by bow    (+20).  Wrong action: −3, −10 energy.
  SmartEntity  — defeated by sword  (+20).  Wrong action: −3, −10 energy.

All defeated entities are permanently removed from the maze.

Energy system
-------------
  Robot starts each episode with ENERGY_MAX energy.
  Energy depletes through actions:
    move step        : −1
    correct combat   : −2
    wrong combat     : −10
    wasted combat    : −3  (combat action with nothing ahead)
  Reaching energy = 0 truncates the episode (no exit bonus earned).
  Reaching the exit awards: +100 + (energy / ENERGY_MAX) × 20.
  Maximum possible score per episode: 120.

Prepared-weapon bonus
---------------------
  If an entity was visible at 2-block range before entering 1-block range,
  and the robot then uses the correct combat action, it earns +5 on top of
  the +20 combat reward.  This incentivises reading the 2-block sensor and
  committing to the right weapon before the enemy is adjacent.

  Line-of-sight rule: 2-block detection is blocked if the 1-block cell
  contains a wall or another entity (corner-spawn scenario).

State (17-D, via flatten_observation_entity)
--------------------------------------------
  Dims  0-9 : base navigation state (flatten_observation_v2)
  Dims 10-12: entity at 1 block  — tank / flying / smart
  Dims 13-15: entity at 2 blocks — tank / flying / smart (0 if LoS blocked)
  Dim  16   : energy normalised  — energy / ENERGY_MAX  ∈ [0, 1]
"""

from enum import IntEnum

import numpy as np
import pygame
from gymnasium import spaces

from envs.simple_dungeonworld_env import DungeonMazeEnv
from core.dungeonworld_objects import TankEntity, FlyingEntity, SmartEntity


ENERGY_MAX = 100

# Lazy import — only loaded when use_perception=True
_PerceptionPipeline = None

def _get_perception():
    global _PerceptionPipeline
    if _PerceptionPipeline is None:
        from utils.perception import PerceptionPipeline
        _PerceptionPipeline = PerceptionPipeline
    return _PerceptionPipeline


class EntityActions(IntEnum):
    turn_right    = 0
    turn_left     = 1
    move_forwards = 2
    flee          = 3
    use_bow       = 4
    use_sword     = 5


# Combat table: action → [beats_tank, beats_flying, beats_smart]
_COMBAT_TABLE = {
    EntityActions.flee:      [True,  False, False],
    EntityActions.use_bow:   [False, True,  False],
    EntityActions.use_sword: [False, False, True],
}
_ENTITY_IDX = {'tank': 0, 'flying': 1, 'smart': 2}

_ENTITY_COLOURS = {
    'tank':   (180,  60,  60),
    'flying': ( 60,  60, 180),
    'smart':  ( 60, 160,  60),
}

# Direction vectors: direction index → (dx, dy)
_DIR = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


class DungeonMazeEntityEnv(DungeonMazeEnv):
    """DungeonMaze with entities, energy, and 2-block advance detection."""

    def __init__(self, render_mode=None, grid_size=12, use_shaping=False,
                 entity_positions=None, use_perception=False,
                 disable_energy=False, disable_prepared_bonus=False):
        """
        entity_positions : dict or None
            Override spawn positions for specific entities.
            Keys: 'tank', 'flying', 'smart'  Values: (x, y) tuples.
            Example: {'tank': (3, 5), 'flying': (9, 2)}
            Unspecified entities fall back to the automatic placement.

        use_perception : bool
            When True, entity sensors use Task 1 classifiers (KNN at 1 block,
            CNN at 2 blocks) and return continuous cluster probabilities.
            CNN scan costs -5 energy the first time an entity is detected at
            2-block range. When False (default), oracle binary flags are used —
            fast training mode.

        disable_energy : bool
            Ablation flag. When True, all energy costs are removed and the
            episode never truncates from energy depletion. The goal bonus is
            flat +100 (no energy component). Isolates the effect of the energy
            system on agent behaviour.

        disable_prepared_bonus : bool
            Ablation flag. When True, the +5 prepared-weapon bonus is never
            awarded. Isolates whether advance 2-block detection improves policy.
        """
        super().__init__(render_mode=render_mode, grid_size=grid_size,
                         use_shaping=use_shaping)
        self._entity_positions       = entity_positions or {}
        self.use_perception          = use_perception
        self.disable_energy          = disable_energy
        self.disable_prepared_bonus  = disable_prepared_bonus
        self._perception             = _get_perception()() if use_perception else None

        self.action_space = spaces.Discrete(len(EntityActions))

        # Entity sensors — Box [0,1] covers both oracle binary and perception probs
        _sensor_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space.spaces["tank_front"]    = _sensor_space
        self.observation_space.spaces["flying_front"]  = _sensor_space
        self.observation_space.spaces["smart_front"]   = _sensor_space
        self.observation_space.spaces["tank_2block"]   = _sensor_space
        self.observation_space.spaces["flying_2block"] = _sensor_space
        self.observation_space.spaces["smart_2block"]  = _sensor_space
        # Energy
        self.observation_space.spaces["energy"] = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.energy   = float(ENERGY_MAX)
        self._prepared: set = set()      # entity types seen at 2 blocks
        self._scanned_2block: dict = {}  # entity_type → [p_tank, p_flying, p_smart]
        self._scanned_1block: dict = {}  # entity_type → [p_tank, p_flying, p_smart]

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._place_entities()
        self.energy          = float(ENERGY_MAX)
        self._prepared       = set()
        self._scanned_2block = {}
        self._scanned_1block = {}
        obs = self.get_observations()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    # ─── Entity placement ─────────────────────────────────────────────────────

    def _place_entities(self):
        walkable = []
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                cell = self.maze.get_cell_item(x, y)
                pos  = np.array([x, y])
                if (cell is None
                        and not np.array_equal(pos, self.robot_position)
                        and not np.array_equal(pos, self.target_position)
                        and np.linalg.norm(pos - self.robot_position) > 3.0):
                    walkable.append((x, y))

        walkable.sort()
        n = len(walkable)

        auto = {
            'tank':   walkable[n // 4],
            'flying': walkable[n // 2],
            'smart':  walkable[3 * n // 4],
        }

        for etype, EntityClass in [
            ('tank',   TankEntity),
            ('flying', FlyingEntity),
            ('smart',  SmartEntity),
        ]:
            x, y = self._entity_positions.get(etype, auto[etype])
            self.maze.add_cell_item(x, y, EntityClass(np.array([x, y])))

    # ─── Sensing ──────────────────────────────────────────────────────────────

    def _entity_at_distance(self, n: int):
        """
        Return entity type n cells ahead, or None.
        Returns None if any cell between robot and target is a wall or entity
        (line-of-sight blocked).
        """
        rx, ry = int(self.robot_position[0]), int(self.robot_position[1])
        dx, dy = _DIR[self.robot_direction]

        for step in range(1, n + 1):
            nx, ny = rx + dx * step, ry + dy * step
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                return None
            cell = self.maze.get_cell_item(nx, ny)
            if cell is None:
                continue
            if cell.type == "wall":
                return None          # LoS blocked by wall
            if cell.type in _ENTITY_IDX:
                return cell.type if step == n else None  # entity blocks LoS before n

        return None

    def get_observations(self):
        obs = super().get_observations()

        entity_1 = self._entity_at_distance(1)
        entity_2 = self._entity_at_distance(2)

        if self.use_perception and self._perception is not None:
            # 1-block: KNN on full-brightness sprite
            if entity_1 is not None:
                if entity_1 not in self._scanned_1block:
                    self._scanned_1block[entity_1] = \
                        self._perception.perceive_1block(entity_1)
                p1 = self._scanned_1block[entity_1]
            else:
                p1 = np.zeros(3, dtype=np.float32)

            # 2-block: read from cache only (scan + energy charge happen in step())
            if entity_2 is not None and entity_2 in self._scanned_2block:
                p2 = self._scanned_2block[entity_2]
            else:
                p2 = np.zeros(3, dtype=np.float32)

            obs["tank_front"]    = float(p1[0])
            obs["flying_front"]  = float(p1[1])
            obs["smart_front"]   = float(p1[2])
            obs["tank_2block"]   = float(p2[0])
            obs["flying_2block"] = float(p2[1])
            obs["smart_2block"]  = float(p2[2])
        else:
            # Oracle binary flags — fast training mode
            obs["tank_front"]    = int(entity_1 == 'tank')
            obs["flying_front"]  = int(entity_1 == 'flying')
            obs["smart_front"]   = int(entity_1 == 'smart')
            obs["tank_2block"]   = int(entity_2 == 'tank')
            obs["flying_2block"] = int(entity_2 == 'flying')
            obs["smart_2block"]  = int(entity_2 == 'smart')

        obs["energy"] = np.array([self.energy / ENERGY_MAX], dtype=np.float32)
        return obs

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        # Update prepared set before any movement changes position
        entity_2 = self._entity_at_distance(2)
        if entity_2 is not None:
            self._prepared.add(entity_2)
            # CNN scan: charge energy AND store result on first detection
            if self.use_perception and self._perception and entity_2 not in self._scanned_2block:
                if not self.disable_energy:
                    self.energy = max(0.0, self.energy - 5.0)
                self._scanned_2block[entity_2] = self._perception.perceive_2block(entity_2)

        if action in (EntityActions.flee, EntityActions.use_bow,
                      EntityActions.use_sword):
            return self._combat_step(action)

        # Movement step
        obs, reward, terminated, truncated, info = super().step(action)
        if not self.disable_energy:
            self.energy -= 1.0
            if self.energy <= 0:
                self.energy = 0.0
                truncated   = True

        # Refresh obs with new entity sensors and energy
        obs = self.get_observations()
        return obs, reward, terminated, truncated, info

    def _combat_step(self, action):
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        entity_1 = self._entity_at_distance(1)

        if entity_1 is None:
            reward = -0.5
            if not self.disable_energy:
                self.energy = max(0.0, self.energy - 3.0)
        else:
            robot_wins = _COMBAT_TABLE[action][_ENTITY_IDX[entity_1]]

            if robot_wins:
                # Permanently remove entity
                front = self.get_robot_front_pos()
                self.maze.grid[
                    int(front[1]) * self.grid_size + int(front[0])
                ] = None
                reward = 20.0
                if not self.disable_energy:
                    self.energy = max(0.0, self.energy - 2.0)

                # Prepared bonus: robot detected entity at 2 blocks before combat
                if not self.disable_prepared_bonus and entity_1 in self._prepared:
                    reward += 5.0
                self._prepared.discard(entity_1)
            else:
                reward = -3.0
                if not self.disable_energy:
                    self.energy = max(0.0, self.energy - 10.0)

        if not self.disable_energy and self.energy <= 0:
            truncated = True

        # Loop-detection penalty
        self.last_positions.append(tuple(self.robot_position))
        if len(self.last_positions) > 5:
            self.last_positions.pop(0)
        if self.last_positions.count(self.last_positions[-1]) >= 3:
            reward -= 0.5

        terminated = bool(np.array_equal(self.robot_position, self.target_position))
        if terminated:
            energy_bonus = 0.0 if self.disable_energy else (self.energy / ENERGY_MAX) * 20.0
            reward = 100.0 + energy_bonus

        observation = self.get_observations()
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, {}

    # ─── Rendering ────────────────────────────────────────────────────────────

    def _render_frame(self):
        result = super()._render_frame()

        if self.render_mode == "human" and self.window is not None:
            pix    = self.window_size / self.grid_size
            canvas = pygame.Surface((self.window_size, self.window_size),
                                    pygame.SRCALPHA)
            canvas.fill((0, 0, 0, 0))

            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    cell = self.maze.get_cell_item(x, y)
                    if cell is not None and cell.type in _ENTITY_COLOURS:
                        pygame.draw.rect(
                            canvas, _ENTITY_COLOURS[cell.type],
                            pygame.Rect(pix * x, pix * y, pix, pix)
                        )

            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()

        return result
