import pygame
import time
import numpy as np
from fruit_catcher_env import FruitCatacherEnv
from q_model import build_model

#Intiialize pygame
pygame.init()

CELL_SIZE = 40
WIDTH = 5
HEIGHT = 5
SCREEN = pygame.display.set_mode((CELL_SIZE * WIDTH, CELL_SIZE * HEIGHT))
pygame.display.set_caption("Fruit Catcher AI!")

WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)

model = build_model()
model.load_weights("model_weights.weights.h5")

env = FruitCatacherEnv()
state = env.reset()

clock=pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    q_values = model.predict(state.reshape(1, -1), verbose=0)
    action = np.argmax(q_values[0])
    
    state, _, done = env.step(action)

    SCREEN.fill(WHITE)

    pygame.draw.rect(SCREEN,RED, (
        env.fruit_pos[0] * CELL_SIZE,
        env.fruit_pos[1] * CELL_SIZE,
        CELL_SIZE, CELL_SIZE))
    
    pygame.draw.rect(SCREEN, BLUE, (
        env.basket_pos * CELL_SIZE,
        (env.height - 1) * CELL_SIZE,
        CELL_SIZE, CELL_SIZE
    ))

    pygame.display.flip() #updates the screen withe verything drawn
    clock.tick(5)

    if done:
        time.sleep(1)
        state=env.reset()

pygame.quit()