import dezero.functions as F
import dezero.layers as L
import gym
import numpy as np
from dezero import Model, optimizers


class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    agent = Agent()

    action, prob = agent.get_action(state)
    print("action:", action)
    print("prob:", prob)

    G = 100.0
    J = G * F.log(prob)
    print("J:", J)

    J.backward()
