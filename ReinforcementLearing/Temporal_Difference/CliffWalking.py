import numpy
import matplotlib.pyplot as plt

world_height = 4
world_width = 12

epsilon = 0.1
alpha = 0.5

gamma = 1

action_up = 0
action_down = 1
action_left = 2
action_right = 3

actions = [action_up,action_down,action_left,action_right]
actionRewards = numpy.zeros((world_height, world_width, 4))
actionRewards[:, :, :] = -1.0
actionRewards[2, 1:11, action_down] = -100.0
actionRewards[3, 0, action_right] = -100.0

stateActionValues = numpy.zeros((world_height, world_width, 4))
startState = [3, 0]
goalState = [3, 11]

actionDestination = []
for i in range(0, world_height):
    actionDestination.append([])
    for j in range(0, world_width):
        destinaion = dict()
        destinaion[action_up] = [max(i - 1, 0), j]
        destinaion[action_left] = [i, max(j - 1, 0)]
        destinaion[action_right] = [i, min(j + 1, world_width - 1)]
        if i == 2 and 1 <= j <= 10:
            destinaion[action_down] = startState
        else:
            destinaion[action_down] = [min(i + 1, world_height - 1), j]
        actionDestination[-1].append(destinaion)
actionDestination[3][0][action_right] = startState

def chooseAction(state, stateActionValues):
    if numpy.random.binomial(1, epsilon) == 1:
        return numpy.random.choice(actions)
    else:
        values_ = stateActionValues[state[0], state[1], :]
        return numpy.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == numpy.max(values_)])

def sarsa(stateActionValues,expected=False, stepSize=alpha):
    currentState = startState
    currentAction = chooseAction(currentState, stateActionValues)
    rewards = 0.0

    while currentState != goalState:
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        newAction = chooseAction(newState, stateActionValues)
        reward = actionRewards[currentState[0], currentState[1], currentAction]
        rewards += reward
        if not expected:
            valueTarget = stateActionValues[newState[0], newState[1], newAction]
        else:
            # expext sarsa method
            valueTarget = 0.0
            actionValues = stateActionValues[newState[0], newState[1], :]
            bestActions = numpy.argwhere(actionValues == numpy.max(actionValues))
            for action in actions:
                # here the calculation depend on the e-greedy method
                if action in bestActions:
                    # e-greedy 的 非e 情况下
                    valueTarget += (1.0 - epsilon) / len(bestActions) * stateActionValues[newState[0], newState[1], action]
                else:
                    # e-greedy 的 e 情况下随机选择
                    valueTarget += epsilon / len(actions) * stateActionValues[newState[0], newState[1], action]

        valueTarget *= gamma
        stateActionValues[currentState[0], currentState[1], currentAction] += stepSize * (reward + valueTarget - stateActionValues[currentState[0],currentState[1], currentAction])
        currentState = newState
        currentAction = newAction

    return rewards

def qLearning(stateActionValues, stepSize=alpha):
    currentState = startState
    rewards = 0.0
    while currentState != goalState:
        currentAction = chooseAction(currentState, stateActionValues)
        reward = actionRewards[currentState[0], currentState[1], currentAction]
        rewards += reward

        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        stateActionValues[currentState[0], currentState[1], currentAction] += stepSize * (reward + gamma * numpy.max(stateActionValues[newState[0], newState[1], :]) -
                                                                                          stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
    return rewards

def printOptimalPolicy(stateActionValues):
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
    for row in optimalPolicy:
        print(row)

def figure6_5():
    # episodes of each run
    nEpisodes = 500

    # perform 20 independent runs
    runs = 20

    rewardsSarsa = numpy.zeros(nEpisodes)
    rewardsQLearning = numpy.zeros(nEpisodes)
    for run in range(0, runs):
        stateActionValuesSarsa = numpy.copy(stateActionValues)
        stateActionValuesQLearning = numpy.copy(stateActionValues)
        for i in range(0, nEpisodes):
            rewardsSarsa[i] += max(sarsa(stateActionValuesSarsa), -100)
            rewardsQLearning[i] += max(qLearning(stateActionValuesQLearning), -100)

    rewardsSarsa /= runs
    rewardsQLearning /= runs

    # averaging the reward sums from 10 successive episodes
    # 平滑化显示
    averageRange = 10
    smoothedRewardsSarsa = numpy.copy(rewardsSarsa)
    smoothedRewardsQLearning = numpy.copy(rewardsQLearning)
    for i in range(averageRange, nEpisodes):
        smoothedRewardsSarsa[i] = numpy.mean(rewardsSarsa[i - averageRange: i + 1])
        smoothedRewardsQLearning[i] = numpy.mean(rewardsQLearning[i - averageRange: i + 1])

    # display optimal policy
    print('Sarsa Optimal Policy:')
    printOptimalPolicy(stateActionValuesSarsa)
    print('Q-Learning Optimal Policy:')
    printOptimalPolicy(stateActionValuesQLearning)

    # draw reward curves
    plt.figure(1)
    plt.plot(smoothedRewardsSarsa, label='Sarsa')
    plt.plot(smoothedRewardsQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()

def figure6_7():
    stepSizes = numpy.arange(0.1, 1.1, 0.1)
    nEpisodes = 1000
    runs = 10

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5
    methods = range(0, 6)

    performace = numpy.zeros((6, len(stepSizes)))
    for run in range(0, runs):
        for ind, stepSize in zip(range(0, len(stepSizes)), stepSizes):
            stateActionValuesSarsa = numpy.copy(stateActionValues)
            stateActionValuesExpectedSarsa = numpy.copy(stateActionValues)
            stateActionValuesQLearning = numpy.copy(stateActionValues)
            for ep in range(0, nEpisodes):
                print('run:', run, 'step size:', stepSize, 'episode:', ep)

                sarsaReward = sarsa(stateActionValuesSarsa, expected=False, stepSize=stepSize)
                expectedSarsaReward = sarsa(stateActionValuesExpectedSarsa, expected=True, stepSize=stepSize)
                qLearningReward = qLearning(stateActionValuesQLearning, stepSize=stepSize)

                performace[ASY_SARSA, ind] += sarsaReward
                performace[ASY_EXPECTED_SARSA, ind] += expectedSarsaReward
                performace[ASY_QLEARNING, ind] += qLearningReward

                if ep < 100:
                    performace[INT_SARSA, ind] += sarsaReward
                    performace[INT_EXPECTED_SARSA, ind] += expectedSarsaReward
                    performace[INT_QLEARNING, ind] += qLearningReward

    performace[:3, :] /= nEpisodes * runs
    performace[3:, :] /= runs * 100
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']
    plt.figure(2)
    for method, label in zip(methods, labels):
        plt.plot(stepSizes, performace[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()

#figure6_5()
figure6_7()
plt.show()
