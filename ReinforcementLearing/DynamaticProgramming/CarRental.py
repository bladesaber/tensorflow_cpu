import numpy
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

max_cars = 20
max_move_of_car = 5

rental_request_first = 3
rental_return_first = 3

rental_request_second = 4
rental_return_second = 2

discount = 0.9
rental_reward = 10
moving_cost = 2

# 当前policy
policy = numpy.zeros((max_cars+1 , max_cars+1))

# 当前状态值
stateValue = numpy.zeros((max_cars+1, max_cars+1))

actions = numpy.arange(-max_move_of_car, max_move_of_car + 1)

# 所有的状态
states = []
# print使用情况
AxisXPrint = []
AxisYPrint = []
for i in range(0, max_cars + 1):
    for j in range(0, max_cars + 1):
        AxisXPrint.append(i)
        AxisYPrint.append(j)
        states.append([i, j])

poissonBackup = dict()
def poisson(n, lam):
    global poissonBackup
    key = n * 10 + lam
    if key not in poissonBackup.keys():
        poissonBackup[key] = math.exp(-lam) * pow(lam, n) / math.factorial(n)
    return poissonBackup[key]

POISSON_UP_BOUND = 10
# @state: [# of cars in first location, # of cars in second location]
# @action: 如果将汽车从第一地点移动到第二位置地点action为正 ,
#          如果将汽车从第二地点移动到第一位置地点action为负
# @stateValue: state value 矩阵
def expectedReturn(state, action, stateValue):
    # 初始化回报
    returns = 0.0

    # 移动车辆的消耗
    returns -= moving_cost * abs(action)

    for rentalRequestFirstLoc in range(0, POISSON_UP_BOUND):
        for rentalRequestSecondLoc in range(0, POISSON_UP_BOUND):

            # 调整车的数量
            numOfCarsFirstLoc = int(min(state[0] - action, max_cars))
            numOfCarsSecondLoc = int(min(state[1] + action, max_cars))

            # 有效的租赁要求应小于实际
            realRentalFirstLoc = min(numOfCarsFirstLoc, rentalRequestFirstLoc)
            realRentalSecondLoc = min(numOfCarsSecondLoc, rentalRequestSecondLoc)

            # 获取租金
            reward = (realRentalFirstLoc + realRentalSecondLoc) * rental_reward
            numOfCarsFirstLoc -= realRentalFirstLoc
            numOfCarsSecondLoc -= realRentalSecondLoc

            # 联合概率分布
            prob = poisson(rentalRequestFirstLoc, rental_request_first) * poisson(rentalRequestSecondLoc, rental_request_second)

            numOfCarsFirstLoc_ = numOfCarsFirstLoc
            numOfCarsSecondLoc_ = numOfCarsSecondLoc
            prob_ = prob
            for returnedCarsFirstLoc in range(0, POISSON_UP_BOUND):
                for returnedCarsSecondLoc in range(0, POISSON_UP_BOUND):
                    numOfCarsFirstLoc = numOfCarsFirstLoc_
                    numOfCarsSecondLoc = numOfCarsSecondLoc_
                    prob = prob_
                    numOfCarsFirstLoc = min(numOfCarsFirstLoc + returnedCarsFirstLoc, max_cars)
                    numOfCarsSecondLoc = min(numOfCarsSecondLoc + returnedCarsSecondLoc, max_cars)
                    prob = poisson(returnedCarsFirstLoc, rental_return_first) * poisson(returnedCarsSecondLoc, rental_return_second) * prob
                    returns += prob * (reward + discount * stateValue[numOfCarsFirstLoc, numOfCarsSecondLoc])
    return returns

def operation():

    # 运算时间过长 , 无法使用

    global policy
    global stateValue

    newStateValue = numpy.zeros((max_cars + 1, max_cars + 1))
    improvePolicy = False
    policyImprovementInd = 0
    while True:
        if improvePolicy == True:
            print('Policy更新', policyImprovementInd)
            policyImprovementInd += 1

            newPolicy = numpy.zeros((max_cars + 1, max_cars + 1))
            for i, j in states:
                actionReturns = []
                for action in actions:
                    # 防止出现 移动的车的数量超出拥有车数量的情况
                    if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                        actionReturns.append(expectedReturn([i, j], action, stateValue))
                    else:
                        actionReturns.append(-float('inf'))
                bestAction = numpy.argmax(actionReturns)
                newPolicy[i, j] = actions[bestAction]

            policyChanges = numpy.sum(newPolicy != policy)
            print('Policy for', policyChanges, 'states changed')
            if policyChanges == 0:
                policy = newPolicy
                break
            policy = newPolicy
            improvePolicy = False

        for i, j in states:
            newStateValue[i, j] = expectedReturn([i, j], policy[i, j], stateValue)
        if numpy.sum(numpy.abs(newStateValue - stateValue)) < 1e-4:
            stateValue[:] = newStateValue
            improvePolicy = True
            continue
        stateValue[:] = newStateValue

# 显示policy/state value矩阵
figureIndex = 0
def prettyPrint(data, labels):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    ax = fig.add_subplot(111, projection='3d')
    AxisZ = []
    for i, j in states:
        AxisZ.append(data[i, j])
    ax.scatter(AxisXPrint, AxisYPrint, AxisZ)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
prettyPrint(policy, ['# of cars in first location', '# of cars in second location', '# of cars to move during night'])
prettyPrint(stateValue, ['# of cars in first location', '# of cars in second location', 'expected returns'])
plt.show()