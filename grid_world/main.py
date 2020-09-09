import numpy as np

# global parameters
WORLD_SIZE = 7

A_STATE = [0, 1]
B_STATE = [0, 3]

A_HAT = [4, 1]
B_HAT = [2, 3]

GAMMA = 0.9 # discount

ACTIONS = [
    [1, 0], # north
    [-1, 0], # south
    [0, -1], # east
    [0, 1] # west
]

# actions equally likely
ACTION_PROB = 0.25

# step function
def step(state, action):

    if state == A_STATE:

        reward = 10
        state_hat = A_HAT

    elif state == B_STATE:

        reward = 5
        state_hat = B_HAT

    else:

        state_hat = ( np.array(state) + np.array(action) ).tolist()

        # verify if outside the world

        x, y = state_hat
        if (x < 0) or (x >= WORLD_SIZE) or (y < 0) or (y >= WORLD_SIZE):

            reward = -1.0
            state_hat = state

        else:

            reward = 0.0

    return state_hat, reward

def equiprobable_random_policy(first):

    print('\n-> Equiprobable Random Policy\n')

    value = np.zeros((WORLD_SIZE, WORLD_SIZE))

    print('\ngrid world: ')
    print(value)

    n = 0
    while True:

        new_value = np.zeros_like(value)

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # consider all the possible actions
                for action in ACTIONS:

                    # get next state, and the reward of the state
                    state_hat, reward = step([i, j], action)
                    next_i, next_j = state_hat

                    # bellman equation
                    # sum over the actions, and update the value of state [i, j]
                    new_value[i, j] += ACTION_PROB * (reward + GAMMA * value[next_i, next_j])

        n += 1

        if first:
            print('\nFirst iteration: \n')
            value = new_value
            break
        else:

            # check for convergence
            if np.sum( np.abs( value - new_value ) ) < 0.00001:
                print('\nconvergence at step: {} \n'.format(n))
                break

            value = new_value

    # return the value function for this policy
    value = np.round(value, 2)
    return value

if __name__ == '__main__':

    print('\nGRID WORLD :)')

    value = equiprobable_random_policy(first=False)
    print(value)
