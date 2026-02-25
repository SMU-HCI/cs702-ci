"""
IVP Demo: Cliff Launch

A robot is launched from a cliff platform. Different initial velocities
lead to completely different trajectories — the same physics ODE integrated
from different initial conditions.

ODE (Newton's 2nd Law):
    x'' = 0           (no horizontal forces in air)
    y'' = -g          (gravity)
    + contact forces   (normal force + friction on surfaces)

pymunk integrates this IVP from each (x0, y0, vx0, vy0) you provide.

Controls:
    Click & drag on the platform  →  launch robot (drag direction = initial velocity)
    R  →  reset
    ESC  →  quit
"""

import math
import pygame
import pymunk

SCREEN_W, SCREEN_H = 900, 650

DARK_BG    = (25,  30,  35)
PLATFORM_C = (55,  90,  55)
PLATFORM_E = (80, 130,  80)   # edge highlight
GROUND_C   = (45,  65,  45)
GROUND_E   = (65,  95,  65)
CLIFF_C    = (200,  70,  70)
WHITE      = (255, 255, 255)
LIGHT_GRAY = (140, 140, 140)
DIM_GRAY   = ( 70,  70,  70)

COLORS = [
    (255, 200,  50),
    ( 80, 200, 255),
    (255, 100, 100),
    (100, 220, 130),
    (200, 130, 255),
    (255, 165,  50),
    (255, 130, 200),
    (130, 255, 230),
]

# Layout (pygame screen coords, y increases downward)
PLATFORM_TOP  = 360    # top surface of left (launch) platform (pygame y)
PLATFORM_X0   = 30
PLATFORM_X1   = 490    # cliff edge x
PLATFORM_H    = 35     # visual thickness
LANDING_X0    = 620    # left edge of landing platform
LANDING_X1    = 875    # right edge
LANDING_TOP   = 420    # slightly lower than launch platform
LANDING_H     = 35
GROUND_Y      = 610    # top of ground (pygame y)
ROBOT_R       = 12
LAUNCH_SCALE  = 4.5    # pixels dragged → px/s velocity
MAX_ROBOTS    = 8
TRAIL_LEN     = 500
GRAVITY       = 900    # px/s²  (pymunk y-up units)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def py2pm(px: float, py: float) -> tuple[float, float]:
    """pygame coords → pymunk coords (y-up)."""
    return px, float(SCREEN_H - py)

def pm2py(px: float, py: float) -> tuple[int, int]:
    """pymunk coords → pygame screen coords."""
    return int(px), int(SCREEN_H - py)


# ── Physics world ─────────────────────────────────────────────────────────────

def build_space() -> pymunk.Space:
    space = pymunk.Space()
    space.gravity = (0, -GRAVITY)

    def seg(x0, y0_py, x1, y1_py, friction=0.8, elasticity=0.3, thickness=2):
        s = pymunk.Segment(
            space.static_body,
            py2pm(x0, y0_py), py2pm(x1, y1_py),
            thickness,
        )
        s.friction = friction
        s.elasticity = elasticity
        space.add(s)
        return s

    # Launch platform surface
    seg(PLATFORM_X0, PLATFORM_TOP, PLATFORM_X1, PLATFORM_TOP, friction=0.7, elasticity=0.05)
    # Landing platform surface
    seg(LANDING_X0, LANDING_TOP, LANDING_X1, LANDING_TOP, friction=0.9, elasticity=0.05)
    # Ground
    seg(-200, GROUND_Y, SCREEN_W + 200, GROUND_Y, friction=0.9, elasticity=0.45)
    # Right wall (keep robots from flying off screen)
    seg(SCREEN_W - 5, GROUND_Y, SCREEN_W - 5, 0, friction=0.5, elasticity=0.4)

    return space


# ── Robot ─────────────────────────────────────────────────────────────────────

class Robot:
    def __init__(self, space: pymunk.Space, x0: float, vx: float, vy_py: float, color, label: str):
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, ROBOT_R)
        self.body = pymunk.Body(mass, moment)
        # Spawn just above platform surface
        spawn_py_y = PLATFORM_TOP - ROBOT_R - 1
        self.body.position = py2pm(x0, spawn_py_y)
        # vy_py: positive = downward in pygame → negate for pymunk y-up
        self.body.velocity = (vx, -vy_py)
        self.shape = pymunk.Circle(self.body, ROBOT_R)
        self.shape.elasticity = 0.35
        self.shape.friction = 0.7
        space.add(self.body, self.shape)
        self.color = color
        self.label = label
        self.trail: list[tuple[int, int]] = []
        self.landed = False
        # Record initial condition for display
        self.ic_text = f"({x0:.0f}, {vx:.0f}, {-vy_py:.0f}) px/s"
        self.spawn_pos = pm2py(*self.body.position)

    def step(self):
        self.trail.append(pm2py(*self.body.position))
        if len(self.trail) > TRAIL_LEN:
            self.trail.pop(0)

    def draw(self, screen):
        n = len(self.trail)
        if n >= 2:
            for i in range(1, n):
                alpha = i / n
                c = tuple(int(ch * (0.08 + 0.92 * alpha)) for ch in self.color)
                pygame.draw.line(screen, c, self.trail[i - 1], self.trail[i], 2)

        cx, cy = pm2py(*self.body.position)
        pygame.draw.circle(screen, self.color, (cx, cy), ROBOT_R)

        vx, vy_pm = self.body.velocity
        spd = math.hypot(vx, vy_pm)
        if spd > 30:
            ex = int(cx + ROBOT_R * vx / spd)
            ey = int(cy - ROBOT_R * vy_pm / spd)   # flip y for screen
            pygame.draw.line(screen, DARK_BG, (cx, cy), (ex, ey), 3)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_arrow(screen, start, end, color):
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < 4:
        return
    pygame.draw.line(screen, color, start, end, 2)
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    h, a = 13, 0.45
    for sign in (a, -a):
        px = int(end[0] - h * (ux * math.cos(sign) - uy * math.sin(sign)))
        py = int(end[1] - h * (uy * math.cos(sign) + ux * math.sin(sign)))
        pygame.draw.line(screen, color, end, (px, py), 2)


LANDING_C  = (55,  75, 110)   # blue-tinted landing platform
LANDING_E  = (80, 120, 170)

def draw_scene(screen, font_sm):
    # Launch platform
    pygame.draw.rect(screen, PLATFORM_C,
                     (PLATFORM_X0, PLATFORM_TOP, PLATFORM_X1 - PLATFORM_X0, PLATFORM_H))
    pygame.draw.line(screen, PLATFORM_E,
                     (PLATFORM_X0, PLATFORM_TOP), (PLATFORM_X1, PLATFORM_TOP), 2)
    # Cliff edge
    pygame.draw.line(screen, CLIFF_C,
                     (PLATFORM_X1, PLATFORM_TOP - 8),
                     (PLATFORM_X1, PLATFORM_TOP + PLATFORM_H + 6), 3)

    # Landing platform
    pygame.draw.rect(screen, LANDING_C,
                     (LANDING_X0, LANDING_TOP, LANDING_X1 - LANDING_X0, LANDING_H))
    pygame.draw.line(screen, LANDING_E,
                     (LANDING_X0, LANDING_TOP), (LANDING_X1, LANDING_TOP), 2)
    # Left edge of landing platform (visual marker)
    pygame.draw.line(screen, LANDING_E,
                     (LANDING_X0, LANDING_TOP - 6),
                     (LANDING_X0, LANDING_TOP + LANDING_H + 4), 2)

    # Ground
    pygame.draw.rect(screen, GROUND_C,
                     (0, GROUND_Y, SCREEN_W, SCREEN_H - GROUND_Y))
    pygame.draw.line(screen, GROUND_E, (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)

    # Labels
    screen.blit(font_sm.render("cliff", True, CLIFF_C), (PLATFORM_X1 + 4, PLATFORM_TOP - 6))
    screen.blit(font_sm.render("LAUNCH", True, LIGHT_GRAY),
                (PLATFORM_X0 + (PLATFORM_X1 - PLATFORM_X0) // 2 - 25, PLATFORM_TOP + 9))
    screen.blit(font_sm.render("LANDING", True, LANDING_E),
                (LANDING_X0 + (LANDING_X1 - LANDING_X0) // 2 - 28, LANDING_TOP + 9))
    screen.blit(font_sm.render("GROUND", True, LIGHT_GRAY), (SCREEN_W // 2 - 25, GROUND_Y + 8))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(screen, clock):
    font_sm = pygame.font.SysFont(None, 21)
    font_md = pygame.font.SysFont(None, 27)

    space = build_space()
    robots: list[Robot] = []
    drag_start: tuple | None = None

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    for r in robots:
                        space.remove(r.body, r.shape)
                    robots.clear()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if PLATFORM_X0 <= mx <= PLATFORM_X1 and my <= PLATFORM_TOP + PLATFORM_H:
                    drag_start = (mx, my)

            if event.type == pygame.MOUSEBUTTONUP:
                if drag_start is not None and len(robots) < MAX_ROBOTS:
                    sx, sy = drag_start
                    ex, ey = event.pos
                    vx  = (ex - sx) * LAUNCH_SCALE
                    vy_py = (ey - sy) * LAUNCH_SCALE   # pygame y-down
                    label = f"IC {len(robots) + 1}"
                    color = COLORS[len(robots) % len(COLORS)]
                    robots.append(Robot(space, float(sx), vx, vy_py, color, label))
                drag_start = None

        # Physics
        space.step(dt)
        for r in robots:
            r.step()
            # Landing detection: robot is on the landing platform surface
            cx, cy = pm2py(*r.body.position)
            if (LANDING_X0 <= cx <= LANDING_X1
                    and LANDING_TOP - ROBOT_R - 5 <= cy <= LANDING_TOP + 5
                    and abs(r.body.velocity[1]) < 80):
                r.landed = True

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill(DARK_BG)
        draw_scene(screen, font_sm)

        for r in robots:
            r.draw(screen)

        # IC labels near spawn points + landed badge
        for r in robots:
            sx, sy = r.spawn_pos
            lbl = font_sm.render(r.label, True, r.color)
            screen.blit(lbl, (sx - lbl.get_width() // 2, sy - ROBOT_R - 18))
            if r.landed:
                cx, cy = pm2py(*r.body.position)
                badge = font_sm.render("LANDED!", True, LANDING_E)
                screen.blit(badge, (cx - badge.get_width() // 2, cy - ROBOT_R - 18))

        # Drag preview
        if drag_start is not None:
            mx, my = pygame.mouse.get_pos()
            preview_color = COLORS[len(robots) % len(COLORS)]
            draw_arrow(screen, drag_start, (mx, my), preview_color)
            vx  = (mx - drag_start[0]) * LAUNCH_SCALE
            vy  = -(my - drag_start[1]) * LAUNCH_SCALE   # physical y (up = positive)
            txt = font_sm.render(f"v₀ = ({vx:+.0f}, {vy:+.0f}) px/s", True, preview_color)
            screen.blit(txt, (drag_start[0] + 14, drag_start[1] - 20))

        # ODE panel (top-right)
        ode_lines = [
            ("ODE — same for every robot:", WHITE),
            ("  x'' = 0  (no horiz. force in air)", LIGHT_GRAY),
            (f"  y'' = −g  (g = {GRAVITY} px/s²)", LIGHT_GRAY),
            ("  + contact forces on surfaces", LIGHT_GRAY),
            ("", WHITE),
            ("Initial condition (you choose):", WHITE),
            ("  x(0),  y(0),  vx(0),  vy(0)", LIGHT_GRAY),
        ]
        ox = SCREEN_W - 310
        for i, (line, color) in enumerate(ode_lines):
            screen.blit(font_sm.render(line, True, color), (ox, 10 + i * 19))

        # IC list
        if robots:
            screen.blit(font_sm.render("Initial velocities set:", True, WHITE), (ox, 160))
            for i, r in enumerate(robots):
                screen.blit(font_sm.render(f"  {r.label}: {r.ic_text}", True, r.color),
                            (ox, 178 + i * 18))

        # Controls
        ctrls = [
            "Drag on platform to launch  |  R reset  |  ESC quit",
            f"Robots: {len(robots)}/{MAX_ROBOTS}",
        ]
        for i, line in enumerate(ctrls):
            screen.blit(font_sm.render(line, True, DIM_GRAY),
                        (10, SCREEN_H - 28 + i * 16))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("IVP Demo: Same ODE, Different Initial Conditions")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    main(screen, clock)
