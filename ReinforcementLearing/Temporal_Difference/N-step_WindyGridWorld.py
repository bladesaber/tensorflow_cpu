import matplotlib.pyplot as plt
import numpy
import random

world_height = 10
world_width = 10

action_up = 0
action_down = 1
action_left = 2
action_right = 3
actionMap = [action_up, action_down, action_left, action_right]

stateActionValues = numpy.zeros((world_height, world_width, 4))
startState = [4, 0]
goalState = [8, 9]

rewardDestination = numpy.zeros((world_height, world_width))
rewardDestination[startState[0],startState[1]] = -1
rewardDestination[goalState[0],goalState[1]] = 1

'''
for i in range(1,world_width-1,2):
    center = random.randint(0,world_height)
    if center == 0:
        range_random = [center,center+1,center+2]
    elif center == world_height-1:
        range_random = [center, center - 1, center - 2]
    else:
        range_random = [center, center + 1, center - 1]
    rewardDestination[:,i] = -1
    for j in range_random:
        rewardDestination[j, i] = 0
'''
epsilon = 0.1
GAMMA = 1

def getReward(state,action,next_state):
    if action == action_up and state[0] == 0:
        return -1
    elif action == action_down and state[0] == world_height-1:
        return -1
    elif action == action_left and state[1] == 0:
        return -1
    elif action == action_right and state[1] == world_width-1:
        return -1
    else:
        return rewardDestination[next_state[0],next_state[1]]

actionDestination = []
for i in range(0, world_height):
    actionDestination.append([])
    for j in range(0, world_width):
        destination = dict()
        destination[action_up] = [max(i - 1, 0), j]
        destination[action_down] = [min(i + 1, world_height - 1), j]
        destination[action_left] = [i , max(j - 1, 0)]
        destination[action_right] = [i , min(j + 1, world_width - 1)]
        actionDestination[-1].append(destination)

def choose_action(currentState):
    if numpy.random.binomial(1, epsilon) == 1:
        currentAction = numpy.random.choice(actionMap)
    else:
        values_ = stateActionValues[currentState[0], currentState[1], :]
        currentAction = numpy.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == numpy.max(values_)])
    return currentAction

# 这种方法有缺陷，有时会形成死循环,alpha 必须很小
def n_step_Sarsa(n, alpha):

    current_state = startState

    states = [current_state]
    rewards = [0]
    actions = []

    currentAction = choose_action(current_state)
    actions.append(currentAction)

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:

        time += 1

        if time < T:

            next_state = actionDestination[current_state[0]][current_state[1]][currentAction]

            nextAction = choose_action(next_state)
            actions.append(nextAction)

            reward = getReward(current_state,currentAction,next_state)
            states.append(next_state)
            rewards.append(reward)

            if next_state == goalState:
                T = time

        update_time = time - n
        if update_time >= 0:
            returns = 0.0

            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += pow(GAMMA, t - update_time - 1) * rewards[t]

            if update_time + n <= T:
                n_state = states[(update_time + n)]
                returns += pow(GAMMA, n) * stateActionValues[n_state[0],n_state[1],actions[update_time+n]]

            # 需要更新的状态
            state_to_update = states[update_time]

            if not state_to_update in goalState:
                stateActionValues[state_to_update[0], state_to_update[1], actions[update_time]] += alpha * (returns - stateActionValues[state_to_update[0],state_to_update[1],actions[update_time]])

        if update_time == T - 1:
            break
        current_state = next_state
        currentAction = nextAction

    return T


def Test(n, alpha):

    values = []

    episodeLimit = 500
    ep = 0

    #episodes = []

    while ep < episodeLimit:
        time = n_step_Sarsa(n,alpha)
        #print('finish at %d' %time)

        #episodes.extend([ep] * time)

        values.append(time)
        ep += 1

    optimalPolicy = []
    for i in range(0, world_height):
        optimalPolicy.append([])
        for j in range(0, world_width):
            if [i, j] == goalState:
                optimalPolicy[-1].append('G ')
                continue
            bestAction = numpy.argmax(stateActionValues[i, j, :])
            if bestAction == action_up:
                optimalPolicy[-1].append('U ')
            elif bestAction == action_down:
                optimalPolicy[-1].append('D ')
            elif bestAction == action_left:
                optimalPolicy[-1].append('L ')
            elif bestAction == action_right:
                optimalPolicy[-1].append('R ')

            if [i, j] == startState:
                word = optimalPolicy[-1][-1]
                word.replace(' ','')
                word += 'S'
                optimalPolicy[-1][-1] = word

            optimalPolicy[-1][-1] += str(rewardDestination[i,j])

    print('Optimal policy is:')
    for row in optimalPolicy:
        print(row)

    #plt.figure()
    #plt.plot(episodes)
    #plt.xlabel('Time steps')
    #plt.ylabel('Episodes')
    #plt.show()

    plt.plot(range(len(values)),values,'-')
    plt.show()

    def go(stateActionValues):
        num = 0
        currentState = startState
        while currentState != goalState:
            bestAction = numpy.argmax(stateActionValues[currentState[0], currentState[1], :])
            if bestAction == action_up:
                print('U')
            elif bestAction == action_down:
                print('D')
            elif bestAction == action_left:
                print('L')
            elif bestAction == action_right:
                print('R')
            currentState = actionDestination[currentState[0]][currentState[1]][bestAction]
            num += 1
            if num>200:
                print('error')
                break

    go(stateActionValues)

Test(3,0.2)

