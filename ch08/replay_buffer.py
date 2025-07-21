import random
from collections import deque

import gym
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = bool


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

    for episode in range(10):
        state, _ = env.reset()
        done = False

        while not done:
            action = 0
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

    env.close()

    state, action, reward, next_state, done = replay_buffer.get_batch()
    print(state.shape)
    print(action.shape)
    print(reward.shape)
    print(next_state.shape)
    print(done.shape)
