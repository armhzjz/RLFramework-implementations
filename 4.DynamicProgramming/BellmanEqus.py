from typing import Dict
import abc
import GridWorld
import numpy


class PolicyEvaluation(metaclass=abc.ABCMeta):
    Discount = float
    theta = float
    reward = float
    State = int
    Actn = int
    ActnProb = float

    def __init__(self,
                    environment: GridWorld.Environment,
                    gamma: Discount = 0.9,
                    policy: Dict[State, Dict[Actn, ActnProb]] or Dict[Actn, ActnProb] or None = None) -> None:
        self._environment = environment
        self._policy = \
            {s: {a: 1. / len(environment.actions) for a in environment.actions.values()} for s in range(environment.num_states)} \
                if policy is None else \
            ({s: policy for s in self._num_states} if policy is Dict[PolicyEvaluation.Actn, PolicyEvaluation.ActnProb] else
                policy)
        self._gamma = gamma
        if policy is Dict[PolicyEvaluation.State, Dict[PolicyEvaluation.Actn, PolicyEvaluation.ActnProb]]:
            assert (len(policy) == int(sum([p for di in policy.values() for p in di.values()]))), \
                "Sum of probabilities of actions does not match number of states"
        elif policy is Dict[PolicyEvaluation.Actn, PolicyEvaluation.ActnProb]:
            assert (1. == sum(p for p in policy.values())), \
                "Sum of probabilities given does not match 1.0"


class BellmanStateValulePolEval(PolicyEvaluation):
    def __init__(self, environment: GridWorld.Environment,
                    gamma: PolicyEvaluation.Discount = 0.9,
                    policy: Dict[PolicyEvaluation.State, Dict[PolicyEvaluation.Actn, PolicyEvaluation.ActnProb]] or Dict[PolicyEvaluation.Actn, PolicyEvaluation.ActnProb] or None = None) -> None:  # noqa: E501
        super().__init__(environment, gamma, policy)

    def __getExpectedReward(self, state: PolicyEvaluation.State) -> PolicyEvaluation.reward:
        actions = self._policy[state]
        expected_reward = 0

        for a, ap in zip(actions.keys(), actions.values()):
            sp, sp_prob, r = self._environment.getPossibleNextStsRew(a)
            state_prime__reward_summatory = 0
            for state_prime, state_prime_p, reward in zip(sp, sp_prob, r):
                state_prime__reward_summatory += state_prime_p * (reward + self._gamma * self._environment.value_states[state_prime])
            expected_reward += ap * state_prime__reward_summatory
        return expected_reward

    def evaluatePolicy(self, theta: PolicyEvaluation.theta = 0.01) -> None:
        while True:
            delta = numpy.array([0] * self._environment.num_states)
            for state in range(self._environment.num_states):
                v = self._environment.value_states[state]
                self._environment.value_states[state] = self.__getExpectedReward(state)
                delta[state] = max(delta[state], abs(v - self._environment.value_states[state]))
            cond = delta <= theta
            if cond.all():
                break
