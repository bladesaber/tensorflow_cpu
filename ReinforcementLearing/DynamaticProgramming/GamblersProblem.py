import numpy
import matplotlib.pyplot as plt

GOAL = 100

# 所有states, 包括0和100
states = numpy.arange(GOAL + 1)

# 头的概率
headProb = 0.4

# 最有策略
policy = numpy.zeros(GOAL + 1)

# state value
stateValue = numpy.zeros(GOAL + 1)
stateValue[GOAL] = 1.0

while True:
    delta = 0.0
    for state in states[1:GOAL]:
        actions = numpy.arange(min(state, GOAL - state) + 1)
        actionReturns = []
        for action in actions:
            actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
        newValue = numpy.max(actionReturns)
        delta += numpy.abs(stateValue[state] - newValue)
        stateValue[state] = newValue
    if delta < 1e-9:
        break

# 计算最优策略
for state in states[1:GOAL]:
    actions = numpy.arange(min(state, GOAL - state) + 1)
    actionReturns = []
    for action in actions:
        actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
    policy[state] = actions[numpy.argmax(actionReturns)]

plt.figure(1)
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.plot(stateValue)
plt.figure(2)
plt.scatter(states, policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.show()
