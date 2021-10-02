from GridWorld import GridWorld, DynamicProgramming


if __name__ == "__main__":

    grid = GridWorld.StateValueGW(
        num_states=16,
        state_row_size=4,
        default_reward=-1.,
        terminal_reward=-1.,
        terminal_states=[0, 15]
    )
    grid.Init(0)

    polEval = DynamicProgramming.PolicyEvaluation(grid, gamma=1.)
    polEval.theta = 0.001
    polEval.evaluatePolicy()
    print("State values calculated are:")
    print(grid.value_states)
