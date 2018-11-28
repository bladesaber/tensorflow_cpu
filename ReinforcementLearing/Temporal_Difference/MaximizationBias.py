import numpy
import matplotlib.pyplot as plt

# state A
state_A = 0

# state B
state_B = 1

# use one terminal state
state_terminal = 2

state_start = state_A

action_right = 0
action_left = 1

epsilson = 0.1
alpha = 0.1
gamma = 1.0

actionsOfB = range(0, 10)

# stateActions = [
#                  state_A:[action_right, action_left],
#                  state_B:[there are 10 actions here]
#                ]
stateActions = [[action_right, action_left], actionsOfB]

# stateActionValues = [state_A,state_B,state_terminal]
stateActionValues = [numpy.zeros(2), numpy.zeros(len(actionsOfB)), numpy.zeros(1)]

actionDestination = [[state_terminal, state_B], [state_terminal] * len(actionsOfB)]

def takeAction(state, action):
    if state == state_A:
        return 0
    return numpy.random.normal(-0.1, 1)

def chooseAction(state, stateActionValues):
    if numpy.random.binomial(1, epsilson) == 1:
        return numpy.random.choice(stateActions[state])
    else:
        values_ = stateActionValues[state]
        return numpy.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == numpy.max(values_)])

def qLearning(stateActionValues, stateActionValues2=None):
    currentState = state_start

    leftCount = 0
    while currentState != state_terminal:
        if stateActionValues2 is None:
            currentAction = chooseAction(currentState, stateActionValues)
        else:
            # derive a action form Q1 and Q2
            currentAction = chooseAction(currentState, [item1 + item2 for item1, item2 in zip(stateActionValues, stateActionValues2)])

        if currentState == state_A and currentAction == action_left:
            leftCount += 1

        reward = takeAction(currentState, currentAction)
        newState = actionDestination[currentState][currentAction]

        if stateActionValues2 is None:
            currentStateActionValues = stateActionValues
            targetValue = numpy.max(currentStateActionValues[newState])
        else:
            if numpy.random.binomial(1, 0.5) == 1:
                currentStateActionValues = stateActionValues
                anotherStateActionValues = stateActionValues2
            else:
                currentStateActionValues = stateActionValues2
                anotherStateActionValues = stateActionValues
            bestAction = numpy.argmax(currentStateActionValues[newState])
            targetValue = anotherStateActionValues[newState][bestAction]

        # Q-Learning update
        currentStateActionValues[currentState][currentAction] +=alpha * (
            reward + gamma * targetValue - currentStateActionValues[currentState][currentAction])
        currentState = newState
    return leftCount

def figure6_8():
    episodes = 300
    leftCountsQ = numpy.zeros(episodes)
    leftCountsDoubleQ = numpy.zeros(episodes)
    runs = 1000
    for run in range(0, runs):
        print('run:', run)

        stateActionValuesQ = [numpy.copy(item) for item in stateActionValues]
        stateActionValuesDoubleQ1 = [numpy.copy(item) for item in stateActionValues]
        stateActionValuesDoubleQ2 = [numpy.copy(item) for item in stateActionValues]

        leftCountsQ_ = [0]
        leftCountsDoubleQ_ = [0]

        for ep in range(0, episodes):
            leftCountsQ_.append(leftCountsQ_[-1] + qLearning(stateActionValuesQ))
            leftCountsDoubleQ_.append(leftCountsDoubleQ_[-1] + qLearning(stateActionValuesDoubleQ1, stateActionValuesDoubleQ2))

        del leftCountsQ_[0]
        del leftCountsDoubleQ_[0]

        leftCountsQ += numpy.asarray(leftCountsQ_, dtype='float') / numpy.arange(1, episodes + 1)
        leftCountsDoubleQ += numpy.asarray(leftCountsDoubleQ_, dtype='float') / numpy.arange(1, episodes + 1)
    leftCountsQ /= runs
    leftCountsDoubleQ /= runs

    plt.figure()
    plt.plot(leftCountsQ, label='Q-Learning')
    plt.plot(leftCountsDoubleQ, label='Double Q-Learning')
    plt.plot(numpy.ones(episodes) * 0.05, label='Optimal')
    plt.xlabel('episodes')
    plt.ylabel('% left actions from A')
    plt.legend()
