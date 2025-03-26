import numpy as np
import matplotlib.pyplot as plt

from utils import Rectangle, Particle, ParticleFilter, BearingsData

domain = Rectangle(center=(0, 0), width=30, height=30)

origin = np.array([0, 0])

bd = BearingsData(
    tbs=[
        (3.0, 135.0),
        (6.0, 110.0)
    ],
    sd_bearing=5.0
)

pf = ParticleFilter(
    n_particles=250,
    start_zone_pos=Rectangle(center=(-8, 5), width=4, height=8),
    start_heading_range=(0, 360),
    start_speed_range=(0.9, 1.2),
    sd_heading=1.5,
)

pf.forecast(duration=3.0, dt=0.1)
pf.compute_weights(bd, obs_index=0)
#pf.resample()
#pf.forecast(duration=3.0, dt=0.1)
#pf.compute_weights(bd, obs_index=1)
#pf.resample()
#pf.forecast(duration=1.0, dt=0.1)

# Draw a picture
fig, ax = plt.subplots()
ax.set_xlim(domain.xlim)
ax.set_ylim(domain.ylim)
ax.set_aspect('equal')

bd.plot_origin(ax)
bd.plot_ray(ax, obs_index=0)
#bd.plot_ray(ax, obs_index=1)
pf.plot_particles(ax)

plt.show()