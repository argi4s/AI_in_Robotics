# SEMTM0016 — DungeonMaze World

A reinforcement learning assignment built on a custom Gymnasium maze environment. Three tasks are implemented end-to-end: supervised species classification (Task 1), unsupervised behaviour clustering (Task 2), and reinforcement learning for navigation and combat (Task 3).

---

## Quick start — run the demos

All commands are run from the project root with the virtual environment active.

### Watch the trained agent (recommended starting point)

```bash
# Full pipeline — DQN agent navigates maze, identifies and defeats enemies
# Uses Task 1 classifiers (SVM + CNN) and Task 2 GMM clusters at runtime
python -m replay.replay_entity
```

```bash
# Navigation only — DQN agent solves the maze without entities
python -m replay.replay_dqn
```

### Play manually

```bash
# Control the robot yourself with keyboard — entity combat included
# Controls: W=forward  A=left  D=right  1=flee  2=bow  3=sword  R=reset  Q=quit
python -m manual.manual_entity
```

### Adjust replay speed and settings

Edit **`replay_config.py`** in the project root — no other files need touching:

```python
STEP_DELAY = 0.35   # seconds between moves  (lower = faster)
MAZE_SEED_NAV    = 42   # maze layout for navigation replay
MAZE_SEED_ENTITY = 77   # maze layout for entity replay
GRID_SIZE  = 12         # maze size (must match trained models)
MAX_STEPS  = 300        # step limit per episode
CELL_SIZE  = 52         # pixel size of each maze cell
```

---

## Installation

Requires Python 3.10 or above.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

---

## Project structure

```
├── replay_config.py          ← edit this to customise demos
│
├── replay/
│   ├── replay_entity.py      ← full DQN+entity demo  ★ start here
│   ├── replay_dqn.py         ← navigation-only DQN demo
│   ├── replay_sarsa.py       ← SARSA tabular agent demo
│   └── replay_q_learning.py  ← Q-Learning tabular agent demo
│
├── manual/
│   └── manual_entity.py      ← human-controlled entity maze
│
├── task1/                    ← supervised learning (KNN, SVM, CNN)
├── task2/                    ← unsupervised learning (PCA, K-Means, GMM)
│
├── train/
│   ├── train_dqn.py          ← train navigation DQN (12×12, 3000 episodes)
│   ├── train_dqn_entities.py ← train entity DQN    (12×12, 5000 episodes)
│   ├── run_ablation.py       ← train ablation configs A / B / C
│   ├── eval_entity.py        ← evaluate saved models (no retraining)
│   └── ablation_table.py     ← print full ablation results table
│
├── envs/
│   ├── simple_dungeonworld_env.py   ← base navigation environment
│   └── entity_dungeonworld_env.py   ← navigation + combat + energy
│
├── models/                   ← DQN, DuelingDQN, SARSA, Q-Learning
├── rl/                       ← DQNAgent, ReplayBuffer
├── utils/                    ← state encoders, observation flatteners
├── saved_models/             ← trained model weights (.pth, .pkl)
└── results/                  ← training curves and ablation plots
```

---

## Reproducing training results

**Tabular agents (SARSA + Q-Learning across 5 grid sizes):**
```bash
python -m agent_comparison
```

**Navigation DQN (12×12, Dueling + Double DQN):**
```bash
python -m train.train_dqn
```

**Entity DQN — full pipeline with combat and perception:**
```bash
python -m train.train_dqn_entities
```

**Ablation study (run after the above):**
```bash
python -m train.run_ablation --config A   # vanilla DQN (no dueling)
python -m train.run_ablation --config B   # entity DQN, no energy system
python -m train.run_ablation --config C   # entity DQN, no prepared-weapon bonus
python -m train.eval_entity               # random baseline + generalisation test
python -m train.ablation_table            # print full results table
```

**Task 2 unsupervised clustering:**
```bash
python -m task2.run_pca_clustering
```

---

## Task 2 → Task 3 integration

The GMM fitted in Task 2 runs at agent inference time. When `replay_entity.py` runs:

1. The robot's **CNN** (Task 1) scans entities at 2-block range and returns `[p_tank, p_flying, p_smart]`
2. The robot's **SVM** (Task 1) confirms at 1-block range with a clearer image
3. These probabilities feed directly into the 17-D state vector the DQN receives
4. The DQN chooses the correct combat action: **flee → tank, bow → flying, sword → smart**

No retraining is needed between tasks — the Task 2 pipeline is serialised to `task2/results/gmm_artefacts.pkl` and loaded at runtime.

---

## Environment — MDP summary

**Action space:** `{turn_right, turn_left, move_forwards}` for navigation; adds `{flee, use_bow, use_sword}` in the entity environment.

**Observation space:** robot position, heading, camera view (20×20 greyscale), target position, wall sensors. Entity environment adds entity probability sensors (1-block and 2-block) and remaining energy.

**Rewards (entity environment):**

| Event | Reward | Energy cost |
|---|---|---|
| Move step | — | −1 |
| Correct combat | +20 | −2 |
| Wrong combat | −3 | −10 |
| Wasted combat (no entity) | −0.5 | −3 |
| Prepared-weapon bonus | +5 | — |
| Goal reached | +100 + energy bonus | — |

**Episode ends** when the robot reaches the exit, runs out of energy, or hits the step limit.
