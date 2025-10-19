import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

R = 384_400_000
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-R * 1.2, R * 1.2)
ax.set_ylim(-R * 1.2, R * 1.2)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title('Движение Луны вокруг Земли')

ax.plot(0, 0, 'bo', markersize=12, label='Земля')
moon, = ax.plot([], [], 'ro', markersize=6, label='Луна')
trail, = ax.plot([], [], 'r-', linewidth=1, alpha=0.6)

trail_x, trail_y = [], []

def animate(frame):
    # print(f"Кадр {frame}")
    t = frame * 0.05
    x = R * np.cos(t)
    y = R * np.sin(t)
    moon.set_data([x], [y])
    trail_x.append(x)
    trail_y.append(y)
    trail.set_data(trail_x, trail_y)
    return moon, trail

anim = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=False, repeat=True)
plt.legend()
plt.show()
