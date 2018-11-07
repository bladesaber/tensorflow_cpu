import numpy
import matplotlib.pyplot as plt

world_height = 7
world_width = 10

wind = [0,0,0,1,1,1,2,2,1,0]

action_up = 0
action_down = 1
action_left = 2
action_right = 3
actions = [action_up, action_down, action_left, action_right]

epsilon = 0.1
alpha = 0.5

stateActionValues = numpy.zeros((world_height, world_width, 4))
startState = [3, 0]
goalState = [3, 7]

# actionDestination 用于表述位移
actionDestination = []
for i in range(0, world_height):
    actionDestination.append([])
    for j in range(0, world_width):
        destination = dict()
        destination[action_up] = [max(i - 1 - wind[j], 0), j]
        destination[action_down] = [max(min(i + 1 - wind[j], world_height - 1), 0), j]
        destination[action_left] = [max(i - wind[j], 0), max(j - 1, 0)]
        destination[action_right] = [max(i - wind[j], 0), min(j + 1, world_width - 1)]
        actionDestination[-1].append(destination)

def oneEpisode():
    time = 0
    currentState = startState

    if numpy.random.binomial(1, epsilon) == 1:
        currentAction = numpy.random.choice(actions)
    else:
        values_ = stateActionValues[currentState[0], currentState[1], :]
        currentAction = numpy.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == numpy.max(values_)])

    while currentState != goalState:
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        if numpy.random.binomial(1, epsilon) == 1:
            newAction = numpy.random.choice(actions)
        else:
            values_ = stateActionValues[newState[0], newState[1], :]
            newAction = numpy.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == numpy.max(values_)])

        if newState == goalState:
            reward = 0.0
        else:
            reward = -1.0

        stateActionValues[currentState[0], currentState[1], currentAction] += alpha * (reward + stateActionValues[newState[0], newState[1], newAction] - stateActionValues[currentState[0], currentState[1], currentAction])

        currentState = newState
        currentAction = newAction

        time += 1
    return time

def Test():
    episodeLimit = 500
    ep = 0
    episodes = []
    while ep < episodeLimit:
        time = oneEpisode()
        episodes.extend([ep] * time)
        ep += 1

    plt.figure()
    plt.plot(episodes)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.show()

    optimalPolicy = []
    for i in range(0, world_height):
        optimalPolicy.append([])
        for j in range(0, world_width):
            if [i, j] == goalState:
                optimalPolicy[-1].append('G')
                continue
            bestAction = numpy.argmax(stateActionValues[i, j, :])
            if bestAction == action_up:
                optimalPolicy[-1].append('U')
            elif bestAction == action_down:
                optimalPolicy[-1].append('D')
            elif bestAction == action_left:
                optimalPolicy[-1].append('L')
            elif bestAction == action_right:
                optimalPolicy[-1].append('R')
    print('Optimal policy is:')
    for row in optimalPolicy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in wind]))

Test()

