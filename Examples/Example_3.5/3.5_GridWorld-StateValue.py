from GridWorld import GridWorld, BellmanEqus


if __name__ == "__main__":

    grid = GridWorld.StateValueGW(
        num_states=25,
        state_row_size=5,
        terminal_states=[],
        irregular_transitions={
            0: {0: (0, -1.), 3: (0, -1.)},
            1: {0: (21, 10.), 1: (21, 10.), 2: (21, 10.), 3: (21, 10.)},
            2: {0: (2, -1.)},
            3: {0: (13, 5.), 1: (13, 5.), 2: (13, 5.), 3: (13, 5.)},
            4: {0: (4, -1.), 2: (4, -1.)},
            5: {3: (5, -1.)},
            9: {2: (9, -1.)},
            10: {3: (10, -1.)},
            14: {2: (14, -1.)},
            15: {3: (15, -1.)},
            19: {2: (19, -1.)},
            20: {1: (20, -1.), 3: (20, -1.)},
            21: {1: (21, -1.)},
            22: {1: (22, -1.)},
            23: {1: (23, -1.)},
            24: {1: (24, -1.), 2: (24, -1.)}
        },
    )
    grid.Init(0)

    polEval = BellmanEqus.BellmanStateValulePolEval(grid, theta=.01)
    polEval.evaluatePolicy()
    print("State values calculated are:")
    print(grid.value_states)
