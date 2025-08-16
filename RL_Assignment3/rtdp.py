from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from gridworld import MDP, State, Action, sample_next_state_and_reward


@dataclass
class LinearDecay:
    start: float
    end: float
    steps: int

    def value(self, t: int) -> float:
        if t <= 0:
            return float(self.start)
        if t >= self.steps:
            return float(self.end)
        frac = t / float(self.steps)
        return float(self.start + frac * (self.end - self.start))


@dataclass
class RTDPConfig:
    gamma: float = 0.95
    episodes: int = 50
    max_steps: int = 1_000
    epsilon_schedule: LinearDecay | None = None


class RTDP:
    def __init__(self, mdp: MDP, cfg: RTDPConfig, rng=None, heuristic=None) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.rng = rng
        self.heuristic = heuristic
        self.V: Dict[State, float] = {}

        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def value(self, s: State) -> float:
        if s not in self.V:
            self.V[s] = float(self.heuristic(s) if self.heuristic else 0.0)
        return self.V[s]

    def bellman_backup(self, s: State) -> float:
        actions = self.mdp.actions(s)
        if not actions:
            self.V[s] = 0.0
            return 0.0

        max_q = float("-inf")
        for a in actions:
            q = 0.0
            for t in self.mdp.transitions(s, a):
                q += t.probability * (t.reward + self.cfg.gamma * self.value(t.next_state))
            if q > max_q:
                max_q = q

        self.V[s] = max_q
        return max_q


    def select_action(self, s: State, epsilon: float) -> Action:
        actions = list(self.mdp.actions(s))
        assert actions
        Q_sa = []
        for a in actions:
            q = 0.0
            for t in self.mdp.transitions(s, a):
                q += t.probability * (t.reward + self.cfg.gamma * self.value(t.next_state))
            Q_sa.append(q)

        # Epsilon-greedy
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        else:
            return actions[Q_sa.index(max(Q_sa))]

    def run(self) -> None:
        episodes = self.cfg.episodes
        for ep in range(episodes):
            s = self.mdp.initial_state()
            steps = 0
            total_reward = 0.0
            epsilon = self.cfg.epsilon_schedule.value(ep) if self.cfg.epsilon_schedule else 0.0

            while not self.mdp.is_terminal(s) and steps < self.cfg.max_steps:
                # Bellman backup for current state
                self.bellman_backup(s)

                # Pick action
                a = self.select_action(s, epsilon)

                # Sample next state and reward
                next_s, reward = sample_next_state_and_reward(self.mdp, s, a, self.rng)

                total_reward += reward
                s = next_s
                steps += 1

            print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward:.2f}")


