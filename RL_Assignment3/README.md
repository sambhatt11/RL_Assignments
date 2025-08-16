# RTDP vs MCTS

RTDP quickly learns an approximate value function and improves its policy with repeated episodes, exploiting the model’s dynamics. It is more deterministic and tends to find the shortest path once it converges.

MCTS explores the action space dynamically, balancing exploration and exploitation with UCT; it can adapt online without a prior value function.

On small GridWorlds, RTDP often converges faster, while MCTS can handle stochastic transitions better and may explore more diverse paths.

RTDP relies on a model and value backup, while MCTS works well even with simulators where transition probabilities are unknown.

