import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Vanilla Deep Q-Network — two hidden layers, raw Q-value output.
    Used for the original 10×10 experiments. Kept intact for backwards
    compatibility with saved checkpoints (best_dqn_model.pth).
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """
    Dueling Network Architecture (Wang et al., 2016).

    The network splits into two streams after the shared feature layers:

        Value stream   V(s)    — scalar: how good is state s overall?
        Advantage stream A(s,a) — per action: how much better is action a?

    Combined:  Q(s, a) = V(s) + [ A(s, a) − mean_a(A(s, a)) ]

    Subtracting the mean advantage makes the decomposition identifiable —
    the network cannot arbitrarily shift V and A in opposite directions
    while keeping Q unchanged. This stabilises training and improves
    generalisation across actions that share similar state values.

    Why it helps for maze navigation
    ---------------------------------
    Many maze states have one clearly best action (e.g. move forward along
    an open corridor). In these states the advantage differences are small
    and the value stream can focus on estimating how close the robot is to
    the exit. The advantage stream then only needs to fine-tune action
    choice at decision points (junctions, dead ends). This decomposition
    leads to faster and more stable learning than a single Q-stream.

    Input:  10-D normalised state from flatten_observation_v2
    Output: Q-value for each of the 3 discrete actions
    """

    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        # Shared feature extractor — same depth and width as vanilla DQN
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # Value stream: single scalar V(s)
        self.fc_value = nn.Linear(256, 1)

        # Advantage stream: one score per action A(s, a)
        self.fc_adv = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = self.fc_value(x)                          # shape: (batch, 1)
        a = self.fc_adv(x)                            # shape: (batch, n_actions)

        # Combine streams; subtract mean advantage for identifiability
        q = v + (a - a.mean(dim=1, keepdim=True))    # shape: (batch, n_actions)
        return q