from typing import Dict, List
from GridWorld import GridWorld
import abc
import numpy


class DynamicProgramming(metaclass=abc.ABCMeta):
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
            ({s: policy for s in self._num_states} if policy is Dict[DynamicProgramming.Actn, DynamicProgramming.ActnProb] else
                policy)
        self._gamma = gamma
        if policy is Dict[DynamicProgramming.State, Dict[DynamicProgramming.Actn, DynamicProgramming.ActnProb]]:
            assert (len(policy) == int(sum([p for di in policy.values() for p in di.values()]))), \
                "Sum of probabilities of actions does not match number of states"
        elif policy is Dict[DynamicProgramming.Actn, DynamicProgramming.ActnProb]:
            assert (1. == sum(p for p in policy.values())), \
                "Sum of probabilities given does not match 1.0"


class PolicyEvaluation(DynamicProgramming):
    def __init__(self, environment: GridWorld.Environment,
                    gamma: DynamicProgramming.Discount = 0.9,
                    theta: DynamicProgramming.theta = 0.01,
                    policy: Dict[DynamicProgramming.State, Dict[DynamicProgramming.Actn, DynamicProgramming.ActnProb]] or Dict[DynamicProgramming.Actn, DynamicProgramming.ActnProb] or None = None) -> None:  # noqa: E501
        super().__init__(environment, gamma, policy)
        self.__policy_Evaluation_theta = theta

    def __getExpectedReward(self, state: DynamicProgramming.State) -> DynamicProgramming.reward:
        actions = self._policy[state]
        expected_reward = 0

        # if the state is a terminal state, there is no next state and
        # the returned reward must zero, given that the episode would be
        # immediately terminated (besides, terminal states have no state value).
        if state in self._environment._terminal_states:
            return DynamicProgramming.reward(0)

        for a, ap in zip(actions.keys(), actions.values()):
            sp, sp_prob, r = self._environment.getPossibleNextStsRew(a, state)
            state_prime__reward_summatory = 0
            for state_prime, state_prime_p, reward in zip(sp, sp_prob, r):
                state_prime__reward_summatory += state_prime_p * (reward + self._gamma * self._environment.value_states[state_prime])
            expected_reward += ap * state_prime__reward_summatory
        return expected_reward

    def evaluatePolicy(self, theta: DynamicProgramming.theta = None) -> None:
        temporal_theta = theta if theta is not None else self.__policy_Evaluation_theta
        delta = numpy.array([numpy.inf] * self._environment.num_states)
        while True:
            for state in range(self._environment.num_states):
                v = self._environment.value_states[state]
                self._environment.value_states[state] = self.__getExpectedReward(state)
                delta[state] = min(delta[state], abs(v - self._environment.value_states[state]))
            cond = delta < temporal_theta
            if cond.all():
                break

    @property
    def theta(self) -> DynamicProgramming.theta:
        return self.__policy_Evaluation_theta

    @theta.setter
    def theta(self, t: DynamicProgramming.theta) -> None:
        self.__policy_Evaluation_theta = t


class EstimationActionValue(PolicyEvaluation):
    def __init__(self, environment: GridWorld.Environment,
                    gamma: DynamicProgramming.Discount = 0.9,
                    theta: DynamicProgramming.theta = 0.01,
                    policy: Dict[DynamicProgramming.State, Dict[DynamicProgramming.Actn, DynamicProgramming.ActnProb]] or Dict[DynamicProgramming.Actn, DynamicProgramming.ActnProb] or None = None) -> None:  # noqa: E501
        super().__init__(environment, gamma, theta, policy)

    def EstimateActionValue(self, value_states: List[GridWorld.Environment.s_sa_value] = None):
        # if value states are given as input, check that its dimentions match
        # the environment's value state
        if value_states is not None:
            assert len(value_states) is len(self._environment.value_states), \
                "Dimentions of given value states and value states of the environment does not match."
            state_values = value_states
        else:
            super(EstimationActionValue, self).evaluatePolicy()
            state_values = self._environment.value_states

        for state in range(self._environment.num_states):
            if self._environment.irregular_transitions is not None and \
                self._environment.current_state in self._environment.irregular_transitions:
                actions = self._environment.irregular_transitions[self._environment.current_state].keys()
            else:
                actions = self._environment.actions.values()
            for action in actions:
                sp, sp_prob, r = self._environment.getPossibleNextStsRew(action, state)
                for state_prime, state_prime_p, reward in zip(sp, sp_prob, r):
                    self._environment.stateAction_values[state][action] += \
                                                                        state_prime_p * (reward + self._gamma * state_values[state_prime])
