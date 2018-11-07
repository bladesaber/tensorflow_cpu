import numpy

world_size = 5
A_pos = [0,1]
A_prime_pos = [4,1]
B_pos = [0,3]
B_prime_pos = [2,3]
discount = 0.9

world = numpy.zeros((world_size,world_size))

actions = ['L','U','R','D']

# 状态转换矩阵
actionProb = []
for i in range(0,world_size):
    actionProb.append([])
    for j in range(0,world_size):
        actionProb[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))

# nextState , actionReward 为游戏的奖励矩阵
nextState = []
actionReward = []
for i in range(0,world_size):
    nextState.append([])
    actionReward.append([])
    for j in range(0,world_size):
        next = dict()
        reward = dict()
        if i == 0:
            next['U'] = [i,j]
            reward['U'] = -1.0
        else:
            next['U'] = [i-1,j]
            reward['U'] = 0.0

        if i == world_size - 1:
            next['D'] = [i,j]
            reward['D'] = -1.0
        else:
            next['D'] = [i+1,j]
            reward['D'] = 0.0

        if j == 0:
            next['L'] = [i, j]
            reward['L'] = -1.0
        else:
            next['L'] = [i, j - 1]
            reward['L'] = 0.0

        if j == world_size - 1:
            next['R'] = [i, j]
            reward['R'] = -1.0
        else:
            next['R'] = [i, j + 1]
            reward['R'] = 0.0

        if [i, j] == A_pos:
            next['L'] = next['R'] = next['D'] = next['U'] = A_prime_pos
            reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0

        if [i, j] == B_pos:
            next['L'] = next['R'] = next['D'] = next['U'] = B_prime_pos
            reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0

        nextState[i].append(next)
        actionReward[i].append(reward)

while True:
    newWorld = numpy.zeros((world_size,world_size))
    for i in range(0,world_size):
        for j in range(0,world_size):
            for action in actions:
                newPosition = nextState[i][j][action]
                newWorld[i, j] += actionProb[i][j][action] * (actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])
    if numpy.sum(numpy.abs(world - newWorld)) < 1e-4:
        print('Random Policy')
        print(newWorld)
        break
    world = newWorld
#print(world)

world = numpy.zeros((world_size, world_size))
while True:
    newWorld = numpy.zeros((world_size, world_size))
    for i in range(0, world_size):
        for j in range(0, world_size):
            values = []
            for action in actions:
                newPosition = nextState[i][j][action]
                values.append(actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])
            newWorld[i][j] = numpy.max(values)
    if numpy.sum(numpy.abs(world - newWorld)) < 1e-4:
        print('Optimal Policy')
        print(newWorld)
        break
    world = newWorld

print(world)