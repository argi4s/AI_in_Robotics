import numpy as np

class SARSAAgent:
    def __init__(self, action_size):
        self.Q = {}
        self.action_size = action_size

    def get_q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_size)
        return self.Q[state]

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.get_q(state))

    def update(self, state, action, reward, next_state, next_action, alpha, gamma):
        q_state = self.get_q(state)
        q_next = self.get_q(next_state)

        # SARSA update (on-policy)
        q_state[action] += alpha * (
            reward + gamma * q_next[next_action] - q_state[action]
        )