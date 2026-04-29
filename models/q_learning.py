import numpy as np

class QLearningAgent:
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

    def update(self, state, action, reward, next_state, alpha, gamma):
        q_state = self.get_q(state)
        q_next = self.get_q(next_state)

        best_next = np.max(q_next)

        q_state[action] += alpha * (
            reward + gamma * best_next - q_state[action]
        )