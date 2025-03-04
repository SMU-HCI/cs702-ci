import pygame
import numpy as np
from skimage.color import hsv2rgb, rgb2hsv
from scipy.stats import multivariate_normal, norm
from dataclasses import dataclass


@dataclass
class Particle:
    x: np.ndarray
    weight: float


pygame.init()

# Define the window size
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

# Set up the window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Moving Circle")

# Parameters for drawing
circle_radius = 15
circle_x = 0
circle_y = WINDOW_HEIGHT // 2
circle_speed = 1

WHITE = (240, 240, 240)
LIGHT_GRAY = (200, 200, 200)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)
BLUE = (0, 153, 181)
YELLOW = (229, 208, 12)
FRAME_RATE = 60  # 60 frames per second

# Initial values
x = np.array([[250, 250, 0]]).T
landmark = np.array([[100, 400]]).T

NP = 100
particles = [Particle(np.copy(x), 1 / NP) for i in range(NP)]


def calc_input() -> np.ndarray:
    v = 50.0  # [1/s]
    yaw_rate = 0.50  # [rad/s]
    return np.array([[v, yaw_rate]]).T


def motion_model(x: np.ndarray, u: np.ndarray, fps=FRAME_RATE) -> np.ndarray:
    A = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
    B = np.array([[np.cos(x[2, 0]), 0],
                  [np.sin(x[2, 0]), 0],
                  [0.0, 1]])
    x = np.dot(A, x) + (1 / fps) * np.dot(B, u)
    return x


def draw_target(x: np.ndarray) -> None:
    circle_radius = 10
    line_length = circle_radius
    cx, cy, heading = float(x[0, 0]), float(x[1, 0]), x[2, 0]
    line_x = cx + line_length * np.cos(heading)
    line_y = cy + line_length * np.sin(heading)
    pygame.draw.circle(window, WHITE, (cx, WINDOW_HEIGHT - cy), circle_radius, 2)
    pygame.draw.line(window, WHITE, (cx, WINDOW_HEIGHT - cy), (line_x, WINDOW_HEIGHT - line_y), 2)


def draw_particles(particles: [Particle]) -> None:
    radius = 8
    h, s, v = rgb2hsv(np.array(YELLOW) / 255)

    particles = sorted(particles, key=lambda p: p.weight, reverse=False)
    max_weight = np.max([p.weight for p in particles])
    for particle in particles:
        v_ = v * (0.50 + 0.50 * particle.weight / max_weight)
        rgb = (hsv2rgb(np.array([h, s, v_])) * 255).astype(np.int16)
        cx, cy, heading = float(particle.x[0, 0]), float(particle.x[1, 0]), particle.x[2, 0]
        pygame.draw.circle(window, rgb, (cx, WINDOW_HEIGHT - cy), radius, 2)


def draw_landmarks(landmarks: [np.ndarray]) -> None:
    for landmark in landmarks:
        lx, ly = float(landmark[0, 0]), float(landmark[1, 0])
        pygame.draw.line(window, LIGHT_GRAY, (lx, WINDOW_HEIGHT - (ly - 5)), (lx, WINDOW_HEIGHT - (ly + 5)), 2)
        pygame.draw.line(window, LIGHT_GRAY, (lx - 5, WINDOW_HEIGHT - ly), (lx + 5, WINDOW_HEIGHT - ly), 2)


def noised_input(u: np.ndarray, fps=FRAME_RATE) -> np.ndarray:
    d_vv, d_vw, d_wv, d_ww, = 5.0, 0.3, 3, 2.0
    cov = np.diag([d_vv ** 2, d_vw ** 2, d_wv ** 2, d_ww ** 2])
    pdf = multivariate_normal(cov=cov)
    noise = pdf.rvs()

    uv = u[0, 0] + noise[0] * np.sqrt(np.abs(u[0, 0]) / fps) + noise[1] * np.sqrt(np.abs(u[1, 0]) / fps)
    uw = u[1, 0] + noise[2] * np.sqrt(np.abs(u[0, 0]) / fps) + noise[3] * np.sqrt(np.abs(u[1, 0]) / fps)
    return np.array([[uv, uw]]).T


def observe(x: np.ndarray, l: np.ndarray):
    dx = l[0, 0] - x[0, 0]
    return dx


def likelihood(observation: float, estimation: float):
    # !!! Implement this! !!!
    return 0


def resample(particles: [Particle]) -> [Particle]:
    prob = [p.weight for p in particles]
    pi = np.random.choice(NP, NP, p=prob)
    new_particles = [Particle(particles[i].x, 1/NP) for i in pi]
    return new_particles


clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    # Move the circle
    u = calc_input()
    x_new = motion_model(x, u)
    x = x_new

    # Move the particles
    particles_new = []
    for particle in particles:
        px = particle.x
        pu = calc_input()
        pu = noised_input(pu)
        px_new = motion_model(px, pu)
        particles_new.append(Particle(px_new, particle.weight))
    particles = particles_new

    # Observation
    observation = observe(x, landmark)

    for particle in particles:
        estimation = observe(particle.x, landmark)
        weight = likelihood(observation, estimation)
        particle.weight *= weight + 1e-10

    # Normalize weight
    total_weight = sum([p.weight for p in particles])
    for particle in particles:
        particle.weight /= total_weight

    # Resampling
    particles = resample(particles)

    # Render
    window.fill(DARK_GRAY)
    draw_landmarks([landmark])
    draw_particles(particles)
    draw_target(x)

    pygame.display.update()
    clock.tick(FRAME_RATE)