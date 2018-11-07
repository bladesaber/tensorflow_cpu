from ReinforcementLearing.GoBang.GoBangSystem import GoBang
import ReinforcementLearing.GoBang.GoBangSystem as GoBangSystem
import copy
import math
import random

class GoBangAI(object):
    directions = [
        [(-1, 0), (1, 0)],
        [(0, -1), (0, 1)],
        [(-1, 1), (1, -1)],
        [(-1, -1), (1, 1)]
    ]

    def __init__(self,goBang,state):
        self.goBang = goBang
        self.state = state
        if self.state == GoBangSystem.ChessboarState.BLACK:
            self.ver_state = GoBangSystem.ChessboarState.WHITE
        else:
            self.ver_state = GoBangSystem.ChessboarState.BLACK

    def estimate(self,i,j,chessMap,state):
        direction_value = []

        for axis in self.directions:
            countNum = 1
            side = []
            for x_direction,y_direction in axis:
                tem_X = i + x_direction
                tem_Y = j + y_direction
                while tem_X>-1 and tem_X<GoBangSystem.N and tem_Y > -1 and tem_Y < GoBangSystem.N:
                    if chessMap[tem_X][tem_Y] == state:
                        countNum += 1
                        tem_X += x_direction
                        tem_Y += y_direction
                    else:
                        side.append(chessMap[tem_X][tem_Y])
                        break
            direction_value.append([countNum,side])

        value = 0
        for countNum,side in direction_value:
            number = len(side)
            if number == 2:
                if countNum == 5:
                    value += 1000
                elif countNum == 4:
                    if(side[0].value + side[1].value == 0):
                        value += 1000
                    elif not GoBangSystem.ChessboarState.EMPTY in side:
                        value += 0
                    else:
                        value += 200
                elif countNum == 3:
                    if (side[0].value + side[1].value == 0):
                        value += 200
                    elif not GoBangSystem.ChessboarState.EMPTY in side:
                        value += 0
                    else:
                        value += 100
                elif countNum == 2:
                    if (side[0].value + side[1].value == 0):
                        value += 80
                    elif not GoBangSystem.ChessboarState.EMPTY in side:
                        value += 0
                    else:
                        value += 60
                elif countNum == 1:
                    if (side[0].value + side[1].value == 0):
                        value += 40
                    elif not GoBangSystem.ChessboarState.EMPTY in side:
                        value += 0
                    else:
                        value += 20
            elif number == 1:
                if countNum == 5:
                    value += 1000
                elif countNum == 4:
                    if(side[0].value == 0):
                        value += 200
                    else:
                        value += 0
                elif countNum == 3:
                    if (side[0].value == 0):
                        value += 100
                    else:
                        value += 0
                elif countNum == 2:
                    if (side[0].value == 0):
                        value += 60
                    else:
                        value += 0
                elif countNum == 1:
                    if (side[0].value== 0):
                        value += 20
                    else:
                        value += 0
        return value

    def predict(self):
        positions = self.goBang.get_Empty()

        value_list = []

        for i,j in positions:
            goBang_copy = copy.deepcopy(self.goBang)
            goBang_copy.set_chessboard_state(i,j,self.state)
            value_1 = self.estimate(i,j,goBang_copy.get_chessMap(),self.state)

            goBang_copy_ver = copy.deepcopy(self.goBang)
            goBang_copy_ver.set_chessboard_state(i, j, self.ver_state)
            value_2 = self.estimate(i, j, goBang_copy_ver.get_chessMap(), self.ver_state)

            value_list.append([value_1 + value_2, [i, j]])

        best_position = []
        MaxValue = -math.inf
        for maxValue,[i,j] in value_list:
            if maxValue > MaxValue:
                best_position.clear()
                best_position.append([i,j])
                MaxValue = maxValue
            elif maxValue == MaxValue:
                best_position.append([i,j])
            else:
                pass

        if len(best_position) > 1:
            return random.choice(best_position)
        else:
            return best_position[0]

