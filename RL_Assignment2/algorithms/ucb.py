import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class UCB(BaseMABAlgorithm):
    def __init__(self, n_arms: int, c: float = 2.0):
        super().__init__(n_arms)
        self.c = c
        
    def select_arm(self) -> int:
        unpulled_arms = np.where(self.pulls == 0)[0]
        if len(unpulled_arms) > 0:
            return int(unpulled_arms[0])
        
        total_pulls = np.sum(self.pulls)
        ucb_values = self.estimates + self.c * np.sqrt(np.log(total_pulls) / self.pulls)
        return int(np.argmax(ucb_values))
