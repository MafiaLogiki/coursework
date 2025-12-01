import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

USE_ROTATING_FRAME = False  # True — Земля и Луна НЕПОДВИЖНЫ (вращающаяся система)


G = 6.67430e-11
m1 = 5.9742e24      # Земля
m2 = 7.36e22        # Луна
R_ORBIT = 384400000 # Среднее расстояние Земля–Луна
T_LUNAR_PERIOD = 27.32 * 86400.0
OMEGA_MOON = 2 * np.pi / T_LUNAR_PERIOD
R_EARTH = 6371000

r2_fixed = np.array([R_ORBIT, 0.0])

def to_rotating_frame(x, y, t, omega=OMEGA_MOON):
    cos_wt = np.cos(omega * t)
    sin_wt = np.sin(omega * t)
    x_rot = x * cos_wt + y * sin_wt
    y_rot = -x * sin_wt + y * cos_wt
    return x_rot, y_rot

def get_r2_moving(t):
    angle = OMEGA_MOON * t
    return np.array([R_ORBIT * np.cos(angle), R_ORBIT * np.sin(angle)])

def dYdt_2D_moving_moon_with_maneuvers(t, Y, maneuvers=None):
    r3 = Y[0:2]
    v3 = Y[2:4]
    r2 = get_r2_moving(t)
    
    r3_mag = np.linalg.norm(r3)
    a1 = -G * m1 * r3 / (r3_mag**3) if r3_mag > 1e-6 else np.zeros(2)
    
    r23_vec = r2 - r3
    r23_mag = np.linalg.norm(r23_vec)
    a2 = G * m2 * r23_vec / (r23_mag**3) if r23_mag > 1e-6 else np.zeros(2)
    
    a_total = a1 + a2

    if maneuvers is not None:
        for man in maneuvers:
            t0 = man["start_time"]
            if t0 <= t <= t0 + 1800:
                dir_vec = man["direction"]
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-8:
                    a_total += (500.0 / 1800.0) * (dir_vec / norm)
                break
    return np.concatenate((v3, a_total))

t0 = 0.0
alpha = np.pi * 95 / 400  # ≈42.75°
V_start = 11100

x3_0 = R_EARTH * np.cos(alpha)
y3_0 = R_EARTH * np.sin(alpha)
vx3_0 = V_start * np.cos(alpha)
vy3_0 = V_start * np.sin(alpha)
Y0 = np.array([x3_0, y3_0, vx3_0, vy3_0])

h = 100.0
total_time = 864000 * 10
total_steps = int(total_time / h)
num_initial_points = 4

def runge_kutta_4th_order_vec(f, t0, Y0, h, num_points):
    t_vals, Y_vals = [t0], [Y0]
    t, Y = t0, Y0
    for _ in range(num_points - 1):
        k1 = h * f(t, Y)
        k2 = h * f(t + h/2, Y + k1/2)
        k3 = h * f(t + h/2, Y + k2/2)
        k4 = h * f(t + h, Y + k3)
        Y = Y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
        t_vals.append(t)
        Y_vals.append(Y)
    return np.array(t_vals), np.array(Y_vals)

def adams_bashforth_4th_order_vec(f, t_init, Y_init, h, total_steps):
    if len(t_init) < 4:
        raise ValueError("Нужно 4 начальные точки.")
    t_vals, Y_vals = list(t_init), list(Y_init)
    F_vals = [f(t, Y) for t, Y in zip(t_init, Y_init)]
    for n in range(3, total_steps):
        Y_next = Y_vals[n] + (h / 24) * (
            55 * F_vals[n] - 59 * F_vals[n-1] + 37 * F_vals[n-2] - 9 * F_vals[n-3]
        )
        t_next = t_vals[n] + h
        t_vals.append(t_next)
        Y_vals.append(Y_next)
        F_vals.append(f(t_next, Y_next))
    return np.array(t_vals), np.array(Y_vals)

print("Выполняется первая симуляция (анализ траектории)...")
dYdt_base = lambda t, Y: dYdt_2D_moving_moon_with_maneuvers(t, Y, maneuvers=None)
t_init, Y_init = runge_kutta_4th_order_vec(dYdt_base, t0, Y0, h, num_initial_points)
t_sol, Y_sol = adams_bashforth_4th_order_vec(dYdt_base, t_init, Y_init, h, total_steps)

# Анализ траектории
r_moon_traj = np.array([get_r2_moving(t) for t in t_sol])
dist_to_moon = np.linalg.norm(Y_sol[:, :2] - r_moon_traj, axis=1)
idx_min = np.argmin(dist_to_moon)
t_closest = t_sol[idx_min]
dist_min = dist_to_moon[idx_min]
print(f"Мин. расстояние до Луны: {dist_min/1000:.1f} км в t = {t_closest/3600:.2f} ч")


maneuvers = None
if dist_min < 5e7:  # ближе 50 000 км
    t_maneuver_start = max(0.0, t_closest - 600)
    idx_man = np.argmin(np.abs(t_sol - t_maneuver_start))
    r3 = Y_sol[idx_man, :2]
    v3 = Y_sol[idx_man, 2:4]
    r2 = get_r2_moving(t_maneuver_start)
    v2 = np.array([-OMEGA_MOON * r2[1], OMEGA_MOON * r2[0]])
    v_rel = v3 - v2
    direction = -v_rel / (np.linalg.norm(v_rel) + 1e-12)
    maneuvers = [{"start_time": t_maneuver_start, "direction": direction}]


if maneuvers is not None:
    print("Выполняется вторая симуляция (с манёвром)...")
    dYdt_with_man = lambda t, Y: dYdt_2D_moving_moon_with_maneuvers(t, Y, maneuvers=maneuvers)
    t_init2, Y_init2 = runge_kutta_4th_order_vec(dYdt_with_man, t0, Y0, h, num_initial_points)
    t_sol2, Y_sol2 = adams_bashforth_4th_order_vec(dYdt_with_man, t_init2, Y_init2, h, total_steps)
    Y_final, t_final = Y_sol2, t_sol2
else:
    Y_final, t_final = Y_sol, t_sol

if USE_ROTATING_FRAME:
    x_inertial = Y_final[:, 0]
    y_inertial = Y_final[:, 1]
    x_anim = []
    y_anim = []
    for i in range(len(t_final)):
        xr, yr = to_rotating_frame(x_inertial[i], y_inertial[i], t_final[i])
        x_anim.append(xr)
        y_anim.append(yr)
    x_anim = np.array(x_anim)
    y_anim = np.array(y_anim)
    t_sol_anim = t_final
else:
    x_anim = Y_final[:, 0]
    y_anim = Y_final[:, 1]
    t_sol_anim = t_final


def animate_solution(x_traj, y_traj, t_sol, use_rotating):
    fig, ax = plt.subplots(figsize=(8, 8))
    VISUAL_RANGE = 1.2 * R_ORBIT
    ax.set_xlim(-VISUAL_RANGE, VISUAL_RANGE)
    ax.set_ylim(-VISUAL_RANGE, VISUAL_RANGE)
    ax.set_aspect('equal')
    
    if use_rotating:
        ax.set_title("Вращающаяся система: Земля и Луна неподвижны")
        ax.plot(0, 0, 'o', color='blue', markersize=10, label='Земля')
        ax.plot(R_ORBIT, 0, 'o', color='gray', markersize=6, label='Луна')
        moon_pt = None
    else:
        ax.set_title("Инерциальная система: Луна движется по орбите")
        ax.plot(0, 0, 'o', color='blue', markersize=10, label='Земля')
        moon_pt, = ax.plot([], [], 'o', color='gray', markersize=6, label='Луна')
    
    sat_pt, = ax.plot([], [], 'ro', markersize=4, label='Аппарат')
    sat_line, = ax.plot([], [], 'r--', linewidth=0.7, alpha=0.8)
    ax.set_xlabel("X (м)"); ax.set_ylabel("Y (м)"); ax.grid(True)
    ax.legend()

    def init():
        out = [sat_pt, sat_line]
        if not use_rotating:
            moon_pt.set_data([], [])
            out.insert(0, moon_pt)
        return out

    def update(i):
        out = []
        if not use_rotating:
            r2 = get_r2_moving(t_sol[i])
            moon_pt.set_data([r2[0]], [r2[1]])
            out.append(moon_pt)
        sat_pt.set_data([x_traj[i]], [y_traj[i]])
        sat_line.set_data(x_traj[:i+1], y_traj[:i+1])
        out.extend([sat_pt, sat_line])
        return out

    frames = range(0, len(t_sol), max(1, len(t_sol) // 1500))
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False, interval=20)
    plt.show()

# Запуск анимации
animate_solution(x_anim, y_anim, t_sol_anim, use_rotating=USE_ROTATING_FRAME)