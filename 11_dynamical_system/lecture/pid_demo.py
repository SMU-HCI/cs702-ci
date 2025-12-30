import pygame
from scipy.stats import norm
from pid import PController, PIController, PIDController

screen_x = 640
screen_y = 480
SCREEN_SIZE = (screen_x, screen_y)
DARK_GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
YELLOW = (255, 191, 0)


def main(screen, clock):
    running = True

    t_sum = 0  # Accumulate time (ms)

    goal_x = 0
    obj_x = 0

    p_control = PController(Kp=0.1)
    pi_control = PIController(Kp=0.1, Ki=0.02)
    pid_control = PIDController(Kp=0.1, Ki=0.02, Kd=0.1)
    u_min, u_max = -50, 50

    while running:
        screen.fill(DARK_GRAY)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                goal_x = event.pos[0]

        # Calculate the control signal
        # u = p_control.calc_input(goal_x, obj_x)
        u = pi_control.calc_input(goal_x, obj_x)
        # u = pid_control.calc_input(goal_x, obj_x)

        u = max(u_min, min(u_max, u))
        disturbance = norm.rvs(scale=0.1)
        u = u + disturbance

        obj_x += u

        # Render the goal
        pygame.draw.line(
            screen, WHITE, (goal_x, screen_y / 2 - 10), (goal_x, screen_y / 2 + 10), 3
        )

        # Render the object
        pygame.draw.circle(screen, YELLOW, (obj_x, screen_y / 2), 10)

        t_sum += clock.get_time()
        pygame.display.flip()
        clock.tick(10)
    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Simple 1D control")

    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()
    main(screen, clock)
