import numpy as np


class RolloutBuffer:
    """
    Buffers a rollout of experiences: states, actions, rewards, dones, etc.
    Automatically converts to arrays and provides batching.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, next_state, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self):
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
            "next_states": np.array(self.next_states, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
        }
