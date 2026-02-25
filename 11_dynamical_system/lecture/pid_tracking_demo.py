import math
import pygame
from pid import PIDController

SCREEN_W, SCREEN_H = 800, 600
DARK_GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
LIGHT_GRAY = (150, 150, 150)
YELLOW = (255, 191, 0)
CYAN = (0, 200, 255)
GREEN = (80, 220, 80)
RED = (220, 80, 80)
ORANGE = (255, 160, 0)

ROBOT_SPEED = 80    # pixels/sec
LOOKAHEAD = 50      # pixels ahead on path


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def find_lookahead(path: list, robot_pos: tuple, lookahead_dist: float):
    """Return (lookahead_point, closest_index) along the path polyline."""
    rx, ry = robot_pos

    # Find closest path point
    closest_idx = min(range(len(path)), key=lambda i: math.hypot(path[i][0] - rx, path[i][1] - ry))

    # Walk forward from closest point until accumulated distance >= lookahead_dist
    accumulated = 0.0
    for i in range(closest_idx, len(path) - 1):
        seg = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if accumulated + seg >= lookahead_dist:
            t = (lookahead_dist - accumulated) / seg if seg > 0 else 0
            lx = path[i][0] + t * (path[i + 1][0] - path[i][0])
            ly = path[i][1] + t * (path[i + 1][1] - path[i][1])
            return (lx, ly), closest_idx
        accumulated += seg

    return path[-1], closest_idx


def draw_robot(screen, x: float, y: float, theta: float, color=YELLOW):
    r = 12
    cx, cy = int(x), int(y)
    pygame.draw.circle(screen, color, (cx, cy), r)
    # Direction indicator
    ex = int(x + r * math.cos(theta))
    ey = int(y + r * math.sin(theta))
    pygame.draw.line(screen, DARK_GRAY, (cx, cy), (ex, ey), 3)


def main(screen, clock):
    font = pygame.font.SysFont(None, 26)

    # state: "idle" | "drawing" | "following" | "done"
    state = "idle"
    path: list[tuple] = []

    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
    pid = PIDController(Kp=3.0, Ki=0.0, Kd=0.3)
    lookahead_pt = None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    state = "idle"
                    path = []
                    lookahead_pt = None

                elif event.key == pygame.K_SPACE and state == "idle" and len(path) >= 2:
                    state = "following"
                    robot_x = float(path[0][0])
                    robot_y = float(path[0][1])
                    dx = path[1][0] - path[0][0]
                    dy = path[1][1] - path[0][1]
                    robot_theta = math.atan2(dy, dx)
                    pid = PIDController(Kp=3.0, Ki=0.0, Kd=0.3)

            if event.type == pygame.MOUSEBUTTONDOWN and state == "idle":
                state = "drawing"
                path = [event.pos]

            if event.type == pygame.MOUSEMOTION and state == "drawing":
                last = path[-1]
                if math.hypot(event.pos[0] - last[0], event.pos[1] - last[1]) > 8:
                    path.append(event.pos)

            if event.type == pygame.MOUSEBUTTONUP and state == "drawing":
                if len(path) >= 2:
                    state = "idle"
                else:
                    state = "idle"
                    path = []

        # --- Robot update ---
        if state == "following":
            lookahead_pt, closest_idx = find_lookahead(path, (robot_x, robot_y), LOOKAHEAD)

            end_dist = math.hypot(path[-1][0] - robot_x, path[-1][1] - robot_y)
            if end_dist < 15 or closest_idx >= len(path) - 2:
                state = "done"
                lookahead_pt = None
            else:
                dx = lookahead_pt[0] - robot_x
                dy = lookahead_pt[1] - robot_y
                target_angle = math.atan2(dy, dx)
                heading_error = normalize_angle(target_angle - robot_theta)

                # PID: drive heading_error to 0
                omega = pid.calc_input(sp=heading_error, pv=0.0, umin=-6.0, umax=6.0)

                # Unicycle kinematics
                robot_x += ROBOT_SPEED * math.cos(robot_theta) * dt
                robot_y += ROBOT_SPEED * math.sin(robot_theta) * dt
                robot_theta = normalize_angle(robot_theta + omega * dt)

        # --- Render ---
        screen.fill(DARK_GRAY)

        # Path
        if len(path) >= 2:
            pygame.draw.lines(screen, CYAN, False, path, 2)

        # Start / end markers
        if len(path) >= 2:
            pygame.draw.circle(screen, GREEN, path[0], 8, 2)
            pygame.draw.circle(screen, RED, path[-1], 8, 2)

        # Lookahead point
        if lookahead_pt:
            pygame.draw.circle(screen, ORANGE, (int(lookahead_pt[0]), int(lookahead_pt[1])), 6)

        # Robot
        if state == "following" or state == "done":
            draw_robot(screen, robot_x, robot_y, robot_theta)
        elif state == "idle" and len(path) >= 2:
            # Ghost robot at start
            draw_robot(screen, float(path[0][0]), float(path[0][1]), 0.0, color=LIGHT_GRAY)

        # HUD
        if state == "idle" and len(path) == 0:
            msg = "Click and drag to draw a path"
        elif state == "drawing":
            msg = "Drawing — release mouse when done"
        elif state == "idle" and len(path) >= 2:
            msg = "Press SPACE to start  |  R to reset"
        elif state == "following":
            msg = "Robot following path...  |  R to reset"
        elif state == "done":
            msg = "Reached goal!  |  R to reset"
        else:
            msg = ""

        screen.blit(font.render(msg, True, WHITE), (10, 10))
        screen.blit(font.render("Orange dot = lookahead target", True, ORANGE), (10, SCREEN_H - 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Path Tracking with PID")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    main(screen, clock)
