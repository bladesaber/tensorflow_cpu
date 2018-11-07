import pygame
from pygame.locals import *
from ReinforcementLearing.GoBang.GoBangSystem import GoBang
import ReinforcementLearing.GoBang.GoBangSystem as GoBangSystem
from ReinforcementLearing.GoBang.GoBangAI import GoBangAI
import random

IMAGE_PATH = 'D:/Git/tensorflow_cpu/ReinforcementLearing/GoBang/'

WIDTH = 540
HEIGHT = 540
MARGIN = 22
GRID = (WIDTH - 2 * MARGIN) / (GoBangSystem.N - 1)
PIECE = 32

class GameRender(object):
    def __init__(self, gobang):
        self.gobang = gobang

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        pygame.display.set_caption('五子棋')

        self.ui_chessboard = pygame.image.load(IMAGE_PATH + 'chessboard.jpg').convert()
        self.ui_piece_black = pygame.image.load(IMAGE_PATH + 'piece_black.png').convert_alpha()
        self.ui_piece_white = pygame.image.load(IMAGE_PATH + 'piece_white.png').convert_alpha()

    def coordinate_transform_map2pixel(self, i, j):
        return MARGIN + i * GRID - PIECE / 2, MARGIN + j * GRID - PIECE / 2

    def coordinate_transform_pixel2map(self, x, y):
        i, j = int((x - MARGIN + PIECE / 2) / GRID), int((y - MARGIN + PIECE / 2) / GRID)
        if i < 0 or i >= GoBangSystem.N or j < 0 or j >= GoBangSystem.N:
            return None, None
        else:
            return i, j

    def draw_chess(self):
        self.screen.blit(self.ui_chessboard, (0, 0))
        for i in range(0, GoBangSystem.N):
            for j in range(0, GoBangSystem.N):
                x, y = self.coordinate_transform_map2pixel(i, j)
                state = self.gobang.get_chessboard_state(i, j)
                if state == GoBangSystem.ChessboarState.BLACK:
                    self.screen.blit(self.ui_piece_black, (x, y))
                elif state == GoBangSystem.ChessboarState.WHITE:
                    self.screen.blit(self.ui_piece_white, (x, y))
                else:
                    pass

    def draw_mouse(self):
        x, y = pygame.mouse.get_pos()
        if self.gobang.currentState == GoBangSystem.ChessboarState.BLACK:
            self.screen.blit(self.ui_piece_black, (x - PIECE / 2, y - PIECE / 2))
        else:
            self.screen.blit(self.ui_piece_white, (x - PIECE / 2, y - PIECE / 2))

    def draw_result(self, result):
        font = pygame.font.SysFont(None,50)
        if result == GoBangSystem.ChessboarState.BLACK:
            tips = "Black Win"
        elif result == GoBangSystem.ChessboarState.WHITE:
            tips = "White Win"
        else:
            tips = "Draw"

        text = font.render(tips, True, (255, 0, 0))
        textRect = text.get_rect()
        #self.screen.blit(text, (WIDTH / 2 - 200, HEIGHT / 2 - 50))
        textRect.centerx = self.screen.get_rect().centerx
        textRect.centery = self.screen.get_rect().centery
        self.screen.blit(text, textRect)

    def one_step(self):
        i, j = None, None
        mouse_button = pygame.mouse.get_pressed()
        if mouse_button[0]:
            x, y = pygame.mouse.get_pos()
            i, j = self.coordinate_transform_pixel2map(x, y)

        if not i is None and not j is None:
            if self.gobang.get_chessboard_state(i, j) != GoBangSystem.ChessboarState.EMPTY:
                return False
            else:
                self.gobang.set_chessboard_state(i, j, self.gobang.currentState)
                return True

        return False

def Start_Game():
    goBang = GoBangSystem.GoBang()
    gameRender = GameRender(goBang)
    goBang.currentState = GoBangSystem.ChessboarState.BLACK

    '''
    if random.randint(0,1) == 1:
        AI = GoBangAI(goBang,GoBangSystem.ChessboarState.BLACK)
        AI_state = GoBangSystem.ChessboarState.BLACK
        person_state = GoBangSystem.ChessboarState.WHITE
    else:
        AI = GoBangAI(goBang, GoBangSystem.ChessboarState.WHITE)
        person_state = GoBangSystem.ChessboarState.BLACK
        AI_state = GoBangSystem.ChessboarState.WHITE
    '''
    AI = GoBangAI(goBang, GoBangSystem.ChessboarState.BLACK)
    AI_state = GoBangSystem.ChessboarState.BLACK
    person_state = GoBangSystem.ChessboarState.WHITE

    result = GoBangSystem.ChessboarState.EMPTY

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif result == GoBangSystem.ChessboarState.EMPTY:
                if goBang.currentState == person_state:
                    if event.type == MOUSEBUTTONDOWN:
                        if gameRender.one_step():
                            result = goBang.count_dimension()
                        else:
                            continue
                        if result != GoBangSystem.ChessboarState.EMPTY:
                            break
                        goBang.currentState = AI_state
                elif goBang.currentState == AI_state:
                    i,j = AI.predict()
                    goBang.set_chessboard_state(i,j,AI_state)
                    result = goBang.count_dimension()
                    if result != GoBangSystem.ChessboarState.EMPTY:
                        break
                    goBang.currentState = person_state

        gameRender.draw_chess()
        gameRender.draw_mouse()

        if result != GoBangSystem.ChessboarState.EMPTY:
            gameRender.draw_result(result)

        pygame.display.update()

Start_Game()