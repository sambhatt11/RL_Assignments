import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class EpsilonGreedy(BaseMABAlgorithm):
    """
    Epsilon-Greedy algorithm
    With probability epsilon: explore (random arm)
    With probability 1-epsilon: exploit (best estimated arm)
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.epsilon = epsilon
        
    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        if np.sum(self.pulls) == 0:
            return 0
        return np.argmax(self.estimates) 