from GridWorld import GridWorld, DynamicProgramming


if __name__ == "__main__":
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
