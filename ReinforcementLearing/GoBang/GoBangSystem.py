from enum import Enum

N = 15

class ChessboarState(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

class GoBang(object):

    directions = [
            [(-1, 0), (1, 0)],
            [(0, -1), (0, 1)],
            [(-1, 1), (1, -1)],
            [(-1, -1), (1, 1)]
        ]

    def __init__(self):
        self.chessMap = [[ChessboarState.EMPTY for j in range(N)] for i in range(N)]
        self.currentI = -1
        self.currentJ = -1
        self.currentState = ChessboarState.EMPTY

    def get_chessMap(self):
        return self.chessMap

    def get_chessboard_state(self,i,j):
        return self.chessMap[i][j]

    def set_chessboard_state(self,i,j,state):
        self.chessMap[i][j] = state
        self.currentI = i
        self.currentJ = j
        self.currentState = state

    def count_dimension(self):
        state = self.chessMap[self.currentI][self.currentJ]

        for axis in self.directions:
            countNum = 1
            for x_direction,y_direction in axis:
                tem_X = self.currentI + x_direction
                tem_Y = self.currentJ + y_direction
                while tem_X>-1 and tem_X<N and tem_Y>-1 and tem_Y<N:
                    if self.chessMap[tem_X][tem_Y] == state:
                        tem_X += x_direction
                        tem_Y += y_direction
                        countNum += 1
                    else:
                        break

            if countNum >= 5:
                return self.currentState

        return ChessboarState.EMPTY

    def get_Empty(self):
        positions = []
        for i in range(N):
            for j in range(N):
                if self.chessMap[i][j] == ChessboarState.EMPTY:
                    positions.append([i,j])
        return positions
