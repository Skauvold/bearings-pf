import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from copy import deepcopy

class Rectangle:
    def __init__(self, center: tuple, width: float, height: float):
        self.center = np.array(center)
        self.width = width
        self.height = height
    
    def draw_uniform(self, n: int):
        x = np.random.uniform(self.xmin, self.xmax, n)
        y = np.random.uniform(self.ymin, self.ymax, n)
        return x, y
    
    @property
    def xmin(self):
        return self.center[0] - self.width / 2
    
    @property
    def xmax(self):
        return self.center[0] + self.width / 2
    
    @property
    def ymin(self):
        return self.center[1] - self.height / 2
    
    @property
    def ymax(self):
        return self.center[1] + self.height / 2

    @property
    def xlim(self):
        return [self.xmin, self.xmax]
    
    @property
    def ylim(self):
        return [self.ymin, self.ymax]


class Particle:
    def __init__(self, position: np.ndarray, heading: float, speed: float):
        self.position_track = [position]
        self.heading_track = [heading]
        self.speed_track = [speed]
    
    def step(self, dt: float, sd_heading: float):
        x1, x2 = self.position
        v1 = self.speed * np.cos(self.heading)
        v2 = self.speed * np.sin(self.heading)
        self.position_track.append(
            (
                x1 + v1 * dt,
                x2 + v2 * dt
            )
        )
        heading_change = np.random.normal(0, sd_heading) * dt
        self.heading_track.append(self.heading + heading_change)
        speed_change = np.random.uniform()
    
    def plot(self, ax: plt.Axes, weight: float):
        cmap = plt.get_cmap("jet")
        particle_color = cmap(weight)
        x = [pos[0] for pos in self.position_track]
        y = [pos[1] for pos in self.position_track]
        ax.plot(x, y, '-', alpha=0.5, color=particle_color)
        ax.plot(self.x, self.y, 'o', markersize=3, color=particle_color)
    
    @property
    def position(self):
        return self.position_track[-1]
    
    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]
    
    @property
    def heading(self):
        return self.heading_track[-1]
    
    @property
    def speed(self):
        return self.speed_track[-1]


class BearingsData:
    def __init__(self, tbs: list, sd_bearing: float):
        self.times = []
        self.bearings = []
        for tb in tbs:
            self.times.append(tb[0])
            self.bearings.append(tb[1])
        self.sd_bearing = sd_bearing
    
    def observation(self, index: int):
        return self.times[index], self.bearings[index]
    
    def time(self, index: int):
        return self.times[index]
    
    def bearing(self, index: int):
        return self.bearings[index]
    
    def likelihood(self, index: int, particle: Particle):
        # Assume current particle time matches times[index]
        # Assume bearing is in degrees
        x, y = particle.position
        particle_bearing = np.arctan2(y, x) * 180 / np.pi
        return stats.norm.pdf(self.bearing(index), particle_bearing, self.sd_bearing)

    def plot_ray(self, ax: plt.Axes, obs_index: int):
        bearing = self.bearing(obs_index)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        xlength = max(xlims) - min(xlims)
        ylength = max(ylims) - min(ylims)
        ray_lenght = np.sqrt(xlength**2 + ylength**2)
        x_ray = [0, ray_lenght * np.cos(bearing * np.pi / 180)]
        y_ray = [0, ray_lenght * np.sin(bearing * np.pi / 180)]
        ax.plot(x_ray, y_ray, '-', color='black', linewidth=0.5)

    def plot_origin(self, ax: plt.Axes):
        ax.plot(0, 0, 's', color='green')


class ParticleFilter:
    def __init__(
            self, n_particles: int,
            start_zone_pos: Rectangle,
            start_heading_range: tuple,
            start_speed_range: tuple,
            sd_heading=1.5
            ):
        self.n_particles = n_particles
        self.particles = [
            Particle(
                position=start_zone_pos.draw_uniform(1),
                heading=np.random.uniform(*start_heading_range),
                speed=np.random.uniform(*start_speed_range)
            ) for _ in range(n_particles)
        ]
        self.sd_heading = sd_heading
        self.particle_weights = np.ones(n_particles) / n_particles
    
    def plot_particles(self, ax: plt.Axes):
        weights_01 = self.particle_weights / np.max(self.particle_weights)

        for weight_01, particle in zip(weights_01, self.particles):
            particle.plot(ax, weight=weight_01)
    
    def forecast(self, duration: float, dt: float):
        n_steps = int(duration / dt)
        for particle in self.particles:
            for _ in range(n_steps):
                particle.step(dt, self.sd_heading)
    
    def compute_weights(self, data: BearingsData, obs_index: int):
        likelihood_values = np.zeros(self.n_particles)
        for particle_index, particle in enumerate(self.particles):
            likelihood_values[particle_index] = data.likelihood(obs_index, particle)
        self.particle_weights = likelihood_values / np.sum(likelihood_values)
        
    def resample(self):
        indices = np.random.choice(range(self.n_particles), size=self.n_particles, p=self.particle_weights)
        self.particles = [deepcopy(self.particles[i]) for i in indices]
        self.particle_weights = np.ones(self.n_particles) / self.n_particles

