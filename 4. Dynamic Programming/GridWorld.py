import abc
import collections
import itertools
from enum import IntEnum
from math import ceil
from typing import Tuple, List, Dict
from random import choices, randint, uniform


class Environment(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'Init') and               # noqa: W504
                callable(subclass.Init) and                 # noqa: W504
                hasattr(subclass, 'GetNext_SR') and         # noqa: W504
                callable(subclass.GetNext_SR) and           # noqa: W504
                hasattr(subclass, 'isEpisodeEnded') and     # noqa: W504
                callable(subclass.isEpisodeEnded) or        # noqa: W504
                NotImplemented)

    class Actions(IntEnum):
        UP = 0
        DOWN = 1
        RIGHT = 2
        LEFT = 3
        TOTAL = 4

    state = int
    reward = float
    t_prob = float
    s_sa_value = float
    state_visits = int

    @abc.abstractmethod
    def GetNext_SR(self, action: Actions) -> Tuple[state, reward]:
        raise NotImplementedError

    @abc.abstractmethod
    def isEpisodeEnded(self):
        raise NotImplementedError

    @abc.abstractmethod
    def Init(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def getPossibleNextStsRew(Actions):
        raise NotImplementedError


class GridWorld(Environment, metaclass=abc.ABCMeta):
    """
        GridWorld class.
        This environment assumes only four possible actions (i.e. UP, DOWN, RIGHT, LEFT).
    """

    Actions = Environment.Actions
    state = Environment.state
    reward = Environment.reward
    t_prob = Environment.t_prob

    def __init__(self, num_states: state = 16,
                    state_row_size: state = 4,
                    default_reward: reward = -1.0,
                    terminal_reward: reward = 0.0,
                    terminal_states: List[state] = [0, 15],
                    irregular_transitions: Dict[state, Dict[Actions, Tuple[state, reward]]] = None,
                    action_overriding_probs: Dict[state, Dict[Actions, t_prob]] or t_prob = None) -> None:
        self._num_states = num_states
        self._state_row_size = state_row_size
        self._default_reward = default_reward
        self._terminal_reward = terminal_reward
        self._terminal_states = terminal_states
        self._irregular_transitions = irregular_transitions
        # a better solution for this type checking must be found here.
        # for the moment I make a very cumbersome and kind of idiotic check
        # just to have it present and obvious as a room of improvement when reading
        # the code.
        assert self._irregular_transitions is None or \
                isinstance(self._irregular_transitions, dict) and \
                isinstance(list(self._irregular_transitions.keys())[0], int) and \
                isinstance(list(self._irregular_transitions.values())[0], dict) and \
                isinstance(list(list(self._irregular_transitions.values())[0].keys())[0], GridWorld.state) and \
                isinstance(list(list(self._irregular_transitions.values())[0].values())[0], Tuple) and \
                isinstance(list(list(self._irregular_transitions.values())[0].values())[0][0], GridWorld.state) and \
                isinstance(list(list(self._irregular_transitions.values())[0].values())[0][1], GridWorld.reward), \
                "Argument irregular_transitions must be either None or a complex dictionary. Take a look at its type hint."
        overr_prob = action_overriding_probs  # axuliar variable with shorter name to avoid E501 .flake8 errors
        self._action_overriding_probs = \
            {s: {a: 0 for a in range(GridWorld.Actions.TOTAL)} for s in range(num_states)} if overr_prob is None else \
            ({s: {a: overr_prob for a in range(GridWorld.Actions.TOTAL)} for s in range(num_states)} if isinstance(overr_prob, GridWorld.t_prob) else  # noqa: E501
                action_overriding_probs)
        if action_overriding_probs is not None:
            circ_q = collections.deque(maxlen=GridWorld.Actions.TOTAL)
            for s, d in zip(self._action_overriding_probs, self._action_overriding_probs.values()):
                for a in d.values():
                    circ_q.append(a)
                if len(circ_q) < GridWorld.Actions.TOTAL:
                    assert 1. >= sum([val for val in d.values()]), \
                        'Sum of transition probabilities of state \'s{}\' are greater than 1.0'.format(s)
                else:
                    for p in range(len(circ_q)):
                        assert 1. >= sum(list(itertools.islice(circ_q, 0, GridWorld.Actions.TOTAL - 1))), \
                            'Sum of any three transition probabilities of state \'s{}\' are greater than 1.0'.format(s)
                        circ_q.rotate()
                circ_q.clear()

    def __check_for_irregular_transitions(self, action: Actions) -> Tuple[state, reward] or None:
        if self._irregular_transitions \
            and self._current_state in self._irregular_transitions \
            and action in self._irregular_transitions[self._current_state]:
                return self._irregular_transitions[self._current_state][action]
        else:
            return None

    def __calculate_next_state(self, action: Actions) -> state:
        state_row = self._current_state // self._state_row_size
        state_position_in_row = self._current_state - (state_row * self._state_row_size)
        if action is GridWorld.Actions.UP:
            return self._current_state - self._state_row_size if state_row > 0 else self._current_state
        elif action is GridWorld.Actions.DOWN:
            num_rows = ceil(float(self._num_states) / float(self._state_row_size))
            return self._current_state + self._state_row_size if state_row < num_rows else self._current_state
        elif action is GridWorld.Actions.RIGHT:
            return self._current_state + 1 if (state_position_in_row < self._state_row_size - 1) and (self._current_state < (self._num_states - 1)) else self._current_state  # noqa: E501
        else:  # action is LEFT
            # state position in row can never be less than zero, being zero always
            # the most left possible place in a row
            return self._current_state - 1 if state_position_in_row > 0 else self._current_state

    def __getActionProbabilities(self, action: Actions) -> Tuple[List[Actions], List[t_prob]]:
        state_overriding_actions = self._action_overriding_probs[self._current_state]  # aux. variable holding the probs per action
        overriding_action_list = [a for a in state_overriding_actions.keys() if a is not action]
        action_overriding_probs = \
            [p for p, a in zip(state_overriding_actions.values(), state_overriding_actions.keys()) if a in overriding_action_list]
        overriding_action_list.append(action)  # append chosen action to the list
        action_overriding_probs.append(1.0 - sum(action_overriding_probs))  # append calculated probability for the chosen action
        return overriding_action_list, action_overriding_probs

    def GetNext_SR(self, action: Actions) -> Tuple[state, reward]:
        overriding_action_list, action_overriding_probs = self.__getActionProbabilities(action)
        action_to_be_taken = choices(overriding_action_list, weights=action_overriding_probs, k=1)[0]

        state_reward_from_irregular_transition = self.__check_for_irregular_transitions(action_to_be_taken)
        if state_reward_from_irregular_transition is not None:
            self._current_state = state_reward_from_irregular_transition[0]  # update the new current state
            return state_reward_from_irregular_transition

        new_s = self.__calculate_next_state(action_to_be_taken)
        self._current_state = new_s  # update the new current state
        return (new_s, self._default_reward) if new_s not in self._terminal_states else (new_s, self._terminal_reward)

    def isEpisodeEnded(self):
        return (self._current_state in self._terminal_states)

    @property
    def actions(self) -> dict:
        return {e.name: action for e, action in zip(GridWorld.Actions, range(GridWorld.Actions.TOTAL))}

    @property
    def num_states(self) -> state:
        return self._num_states

    @property
    def current_state(self) -> state:
        return self._current_state


class StateValueGW(GridWorld):
    def __init__(self, num_states: GridWorld.state = 16,
                    state_row_size: GridWorld.state = 4,
                    default_reward: GridWorld.reward = -1.0,
                    terminal_reward: GridWorld.reward = 0.0,
                    terminal_states: List[GridWorld.state] = [0, 15],
                    irregular_transitions: Dict[GridWorld.state, Dict[GridWorld.Actions, Tuple[GridWorld.state, GridWorld.reward]]] = None,
                    action_overriding_probs: Dict[GridWorld.state, Dict[GridWorld.Actions, GridWorld.t_prob]] or GridWorld. t_prob = None,
                    random_state_values: bool = False) -> None:
        super().__init__(num_states, state_row_size, default_reward, terminal_reward,
                            terminal_states, irregular_transitions, action_overriding_probs)
        self.__random_state_values = random_state_values

    def Init(self, initial_state: GridWorld.state = None):
        self.__value_states = [uniform(1, 100)] * self._num_states if self.__random_state_values else [0] * self._num_states
        self.__state_visits = [0] * self._num_states
        self._current_state = randint(0, self._num_states - 1) if initial_state is None else initial_state

    def getPossibleNextStsRew(self, action: GridWorld.Actions):
        # irregular_transitions: Dict[state, Dict[Actions, Tuple[state, reward]]] = None,
        # action_overriding_probs: Dict[state, Dict[Actions, t_prob]] or t_prob = None)
        actions, action_probs = self.__getActionProbabilities(action)

        if self._irregular_transitions is not None and self._current_state in self._irregular_transitions:
            # get the next state for each action that:
            #   * is present in the actions probability dictionary
            #   * is NOT present in the irregular transitions dictionary
            # for the current state
            # (i.e. the next states of the 'regular' transitions)
            state_prime = [self.__calculate_next_state(action) for action in actions
                            if action not in self._irregular_transitions[self._current_state]
                                and action in self._action_overriding_probs[self._current_state]]
            # get an array with the regular rewards for each one of the state primes...
            rewards = [self._default_reward] * len(state_prime)

            # get all the tuples of action - reward of the actions that:
            #   * are present in the irregular transition dictionary (for the current state)
            #   * are present in the actions probability dictionary (for the current state)
            state_reward_tuple = [t for a in self._irregular_transitions[self._current_state].keys()
                                    for t in self._irregular_transitions[self._current_state].values()
                                    if a in self._action_overriding_probs[self._current_state]]
            # extend the 'state_prime' list with the 'state' elements of the gotten tuples list
            # (i.e. first element, or element 0 of each one of the tuples on list 'state_reward_tuple')
            state_prime.extend(s[0] for s in state_reward_tuple)
            # extend the 'rewards' list with the 'reward' elements of the gotten tuples list
            # (i.e. second element, or element 1 of each one of the tuples on list 'state_reward_tuple')
            rewards.extend(r[1] for r in state_reward_tuple)
        else:
            # get the next states for each action that appears in the action probabilities dictionary
            state_prime = [self.__calculate_next_state(action) for action in actions
                            if action in self._action_overriding_probs[self._current_state]]
            # get an array with the regular rewards for each one of the state primes...
            rewards = [self._default_reward] * len(state_prime)
        # assert there are equal number of next states and action probabilities
        # (or next state probabilities)
        assert len(state_prime) is len(action_probs), "Number of next states is not consistent with number of next state probabilities."
        # assert there are equal number of next states and rewards
        assert len(state_prime) is len(rewards), "Number of next states is not consistent with number of rewards."

        ret_list = [()]
        # state_prime = 
        # action_probs.pop()
        # return (state_prime, state_prime_p, reward)

    @property
    def value_states(self) -> List[Environment.s_sa_value]:
        return self.__value_states

    @value_states.setter
    def value_states(self, val: Environment.s_sa_value) -> None:
        assert type(val) != list, "Value must not be a list AND it must be assigned to an element of property."
        self.__value_states = val

    @property
    def state_visits(self) -> List[Environment.state_visits]:
        return self.__state_visits


class StateActionValueGW(GridWorld):
    def __init__(self, num_states: GridWorld.state = 16,
                    state_row_size: GridWorld.state = 4,
                    default_reward: GridWorld.reward = -1.0,
                    terminal_reward: GridWorld.reward = 0.0,
                    terminal_states: List[GridWorld.state] = [0, 15],
                    irregular_transitions: Dict[GridWorld.state, Dict[GridWorld.Actions, Tuple[GridWorld.state, GridWorld.reward]]] = None,
                    action_overriding_probs: Dict[GridWorld.state, Dict[GridWorld.Actions, GridWorld.t_prob]] or GridWorld. t_prob = None,
                    random_stateaction_values: bool = False) -> None:
        super().__init__(num_states, state_row_size, default_reward, terminal_reward,
                            terminal_states, irregular_transitions, action_overriding_probs)
        self.__random_stateaction_values = random_stateaction_values

    def Init(self, initial_state: GridWorld.state = None):
        if self.__random_stateaction_values:
            self.__stateAction_values = {s: {a: uniform(1, 100) for a in range(GridWorld.Actions.TOTAL)} for s in range(self._num_states)}
        else:
            self.__stateAction_values = {s: {a: 0 for a in range(GridWorld.Actions.TOTAL)} for s in range(self._num_states)}
        self.__state_visits = [0] * self._num_states
        self._current_state = randint(0, self._num_states - 1) if initial_state is None else initial_state

    @property
    def stateAction_values(self) -> Dict[GridWorld.state, Dict[GridWorld.Actions, GridWorld.t_prob]]:
        return self.__stateAction_values

    @property
    def state_visits(self) -> List[Environment.state_visits]:
        return self.__state_visits
