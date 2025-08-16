from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from gridworld import MDP, State, Action, sample_next_state_and_reward


@dataclass
class MCTSConfig:
    gamma: float = 0.95
    c_uct: float = 1.4
    rollouts: int = 200
    max_depth: int = 200


class Node:
    def __init__(self, state: State, parent: Optional[Tuple["Node", Action]] = None) -> None:
        self.state = state
        self.parent = parent
        self.children: Dict[Action, Node] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / float(self.visits)


class MCTS:
    def __init__(self, mdp: MDP, cfg: MCTSConfig, rng=None, heuristic=None) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.rng = rng
        self.heuristic = heuristic
        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def search(self, root_state: State) -> Action:
        root = Node(root_state)
        for _ in range(self.cfg.rollouts):
            # ---- 1. Selection ----
            node = root
            path = [node]
            depth = 0
            while node.children and depth < self.cfg.max_depth:
                # UCT formula
                best_score = -float("inf")
                best_a = None
                for a, child in node.children.items():
                    uct = child.q + self.cfg.c_uct * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))
                    if uct > best_score:
                        best_score = uct
                        best_a = a
                node = node.children[best_a]
                path.append(node)
                depth += 1

            # ---- 2. Expansion ----
            if not self.mdp.is_terminal(node.state):
                for a in self.mdp.actions(node.state):
                    if a not in node.children:
                        next_s, _ = sample_next_state_and_reward(self.mdp, node.state, a, self.rng)
                        node.children[a] = Node(next_s, parent=(node, a))

            # ---- 3. Rollout ----
            rollout_state = node.state
            total_reward = 0.0
            discount = 1.0
            for _ in range(self.cfg.max_depth):
                if self.mdp.is_terminal(rollout_state):
                    break
                actions = list(self.mdp.actions(rollout_state))
                if not actions:
                    break
                a = self.rng.choice(actions)
                rollout_state, reward = sample_next_state_and_reward(self.mdp, rollout_state, a, self.rng)
                total_reward += discount * reward
                discount *= self.cfg.gamma

            # ---- 4. Backpropagation ----
            for n in path:
                n.visits += 1
                n.value_sum += total_reward
        # choose action with most visits
        best_a = None
        best_v = -1
        for a, ch in root.children.items():
            if ch.visits > best_v:
                best_v = ch.visits
                best_a = a
        if best_a is None:
            actions = list(self.mdp.actions(root_state))
            if not actions:
                raise RuntimeError("MCTS on terminal state")
            best_a = actions[0]
        return best_a

