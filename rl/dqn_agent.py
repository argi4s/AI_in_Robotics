import random
import numpy as np
import torch
import torch.nn.functional as F


class DQNAgent:
    """
    DQN agent supporting both vanilla and improved variants.

    Improvements over the original (enabled by default):
      - Double DQN  (use_double=True):
            Decouples action selection from Q-value evaluation to reduce
            overestimation bias. The online network picks the best next action;
            the target network scores it. This prevents the max operator from
            inflating Q-values, which is especially important with dense shaping
            rewards (the maze environment uses +0.5/-0.5 shaping signals).

      - Soft target updates (tau < 1.0):
            Instead of hard-copying weights every N episodes, Polyak averaging
            slowly blends the online network into the target:
                θ_target ← τ·θ_online + (1−τ)·θ_target
            With τ=0.005 the target changes smoothly, reducing the oscillation
            that hard updates cause when early training is noisy (many failed
            episodes on a 16×16 maze).

    The set_networks() interface and all other methods are unchanged so that
    the original train_dqn.py (10×10) continues to work without modification.
    """

    def __init__(self, state_dim, action_dim,
                 gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 use_double=True, tau=0.005):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dim    = action_dim
        self.gamma         = gamma
        self.lr            = lr
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Double DQN flag — set False to revert to vanilla target computation
        self.use_double = use_double

        # Polyak averaging coefficient.  tau=1.0 → hard copy (original behaviour)
        self.tau = tau

        self.q_net      = None
        self.target_net = None
        self.optimizer  = None

    def set_networks(self, q_net, target_net):
        """Attach networks and initialise target weights to match online network."""
        self.q_net      = q_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer  = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def act(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def decay_epsilon(self):
        """
        Multiplicative epsilon decay — call once per episode from the training
        loop so the decay rate is independent of steps-per-episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, batch):
        """One gradient step on a sampled minibatch."""
        states, actions, rewards, next_states, dones = batch

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values: Q(s, a) for the actions actually taken
        q_values = self.q_net(states).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: online network SELECTS the best next action,
                # target network EVALUATES its Q-value.  This breaks the
                # positive feedback loop that causes vanilla DQN to overestimate.
                best_next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(next_states).gather(1, best_next_actions)
            else:
                # Vanilla DQN: same network selects and evaluates
                max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]

            # Zero out Q-values for terminal states (no future reward after done)
            target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """
        Update target network weights.

        Soft update (default, tau < 1.0):
            Polyak averaging: θ_target ← τ·θ_online + (1−τ)·θ_target
            Call every gradient step. tau=0.005 means the target network
            tracks the online network with a lag of ~200 steps.

        Hard update (tau=1.0 or legacy call):
            Full weight copy — preserves the original behaviour.
        """
        if self.tau >= 1.0:
            # Hard copy — original behaviour, kept for backwards compatibility
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            # Polyak averaging — smooth incremental update
            for p_online, p_target in zip(self.q_net.parameters(),
                                          self.target_net.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p_online.data)