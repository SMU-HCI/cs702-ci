import pygame
import numpy as np
from scipy import signal

pygame.init()
width, height = 400, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Scroll Simulation")

WHITE = (255, 255, 255)
RED = (200, 200, 200)
DARK_RED = (100, 100, 100)
BLACK = (0, 0, 0)

a, b, c = 1, 0.5, 0
A = np.array([[0, 1], [-c / a, -b / a]])
B = np.array([[0], [1 / a]])
C = np.array([[1, 1]])
D = np.array([[0]])
SS = signal.StateSpace(A, B, C, D)
t = np.linspace(0, 100, 1000)
u = np.zeros_like(t)
u[10:20] = 3
u[50:60] = -2
u[80:90] = 5
u[100:110] = -3
u[150:160] = 4
u[200:210] = -2
t, y, x = signal.lsim(SS, u, t)

position_scale = 50
speed_scale = 100
time_scale = 100

running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # Current time
    current_time = pygame.time.get_ticks() // time_scale
    if current_time < len(t):
        position = x[current_time, 0] * position_scale

        rect_height = 100
        padding = 10
        # Plot a rectangle with stroke and no fill
        pygame.draw.rect(screen,
                         WHITE,
                         (padding, padding + position, width - 2 * padding, rect_height))

    # Draw a circle to indicate the input
    input_x_origin = width - 50
    input_y_origin = 500
    scale = 10
    input_position = input_y_origin + u[current_time]
    pygame.draw.line(screen,
                     RED,
                     (input_x_origin, input_y_origin),
                     (input_x_origin, int(u[current_time] * scale + input_y_origin)),
                     1)
    pygame.draw.circle(screen,
                       DARK_RED,
                       (input_x_origin, input_y_origin),
                       7)
    pygame.draw.circle(screen,
                       RED,
                       (input_x_origin, int(u[current_time] * scale + input_y_origin)),
                       10)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
