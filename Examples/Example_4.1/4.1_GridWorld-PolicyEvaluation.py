from GridWorld import GridWorld, DynamicProgramming


if __name__ == "__main__":

    grid_stateval = GridWorld.StateValueGW(
        num_states=16,
        state_row_size=4,
        default_reward=-1.,
        terminal_reward=-1.,
        terminal_states=[0, 15]
    )
    grid_stateval.Init(0)

    polEval = DynamicProgramming.PolicyEvaluation(grid_stateval, gamma=1.)
    polEval.theta = 0.001
    polEval.evaluatePolicy()
    print("State values calculated are:")
    print(grid_stateval.value_states)
    print()

    grid_action_value = GridWorld.StateActionValueGW(
        num_states=16,
        state_row_size=4,
        default_reward=-1.,
        terminal_reward=-1.,
        terminal_states=[0, 15]
    )
    grid_action_value.Init(0)
    action_value_estimation = DynamicProgramming.EstimationActionValue(grid_action_value, gamma=1., theta=.001)
    action_value_estimation.EstimateActionValue()
    print("Exercise 4.1:")
    print("\tIn example 4.1, if pi is the equiprobable random policy, what is q_pi(11, down)?")
    print("\t\t q_pi(11, down) = {}".format(grid_action_value.stateAction_values[11][1]))
    print("\tWhat is q_pi(7, down)?")
    print("\t\t q_pi(7, down) = {}".format(grid_action_value.stateAction_values[7][1]))
    print()
    print("Exercise 4.2:")
    print("NOTE:")
    print("NOTE: Since in this implementation state 15 is terminal we enumerate, the here referred to as state 15, as 16 instead")
    print("NOTE: however, in this text state 15 is used to be loyal to the transcript of the book.")
    print("NOTE:")
    print("\tSuppose a new state 15 is added to the gridworld just below state 13, and its actions, left, up, right, and down,")
    print("\ttake the agent to states 12, 13, 14, and 15, respectively. Assume that the t ransitions from the original")
    print("\tstates are unchanged.")
    print("\tWhat, then, is v_pi(16) for the equiprobable random policy?")
    r = grid_stateval.default_reward
    grid_stateval.addIrregularState(
        {
            16: {0: (13, r), 1: (16, r), 2: (14, r), 3: (12, r)},
            12: {1: (12, r)},
            13: {1: (13, r)},
            14: {1: (14, r)}
        }
    )
    grid_stateval.Init(0)
    polEval = DynamicProgramming.PolicyEvaluation(grid_stateval, gamma=1.)
    polEval.theta = 0.001
    polEval.evaluatePolicy()
    print("\t\t v_pi(15) = {}".format(grid_stateval.value_states[16]))
    print()
    print("\tNow suppose the dynamics of state 13 are also changed, such that action down from state 13 takes")
    print("\tthe agent to the new state 15 (state 16 in the real implementation, because in this implementation state 15 is terminal!).")
    print("\tWhat is v_pi(15) for the equiprobable random policy in this case?")
    grid_stateval.addIrregularState(
        {
            16: {0: (13, r), 1: (16, r), 2: (14, r), 3: (12, r)},
            12: {1: (12, r)},
            13: {1: (16, r)},
            14: {1: (14, r)}
        },
        overwrite=True
    )
    polEval = DynamicProgramming.PolicyEvaluation(grid_stateval, gamma=1.)
    polEval.theta = 0.001
    polEval.evaluatePolicy()
    print("\t\t v_pi(15) = {}".format(grid_stateval.value_states[16]))

    grid_action_value = GridWorld.StateActionValueGW(
            num_states=16,
            state_row_size=4,
            default_reward=-1.,
            terminal_reward=-1.,
            terminal_states=[0, 15]
    )
    grid_action_value.Init(0)
    action_value_estimation = DynamicProgramming.PolicyImprovement(grid_action_value, gamma=1., theta=.001)
    action_value_estimation.ImprovePolicy()
    print(action_value_estimation.policy)
