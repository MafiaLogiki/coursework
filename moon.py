import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ================== ВЫБОР РЕЖИМА ==================
USE_ROTATING_FRAME = True  # True — Земля и Луна НЕПОДВИЖНЫ (визуально)
                           # False — Луна движется по орбите (инерциальная система)

# ================== ПАРАМЕТРЫ ==================
G = 6.67430e-11
m1 = 5.9742e24      
m2 = 7.36e22        
R_ORBIT = 384400000 
T_LUNAR_PERIOD = 27.32 * 86400.0  
OMEGA_MOON = 2 * np.pi / T_LUNAR_PERIOD
R_EARTH = 6371000

r2_fixed = np.array([R_ORBIT, 0.0])

def get_r2_moving(t):
    angle = OMEGA_MOON * t
    return np.array([R_ORBIT * np.cos(angle), R_ORBIT * np.sin(angle)])

def to_rotating_frame(x, y, t, omega=OMEGA_MOON):
    """Преобразует инерциальные координаты в систему, вращающуюся с Луной."""
    cos_wt = np.cos(omega * t)
    sin_wt = np.sin(omega * t)
    x_rot = x * cos_wt + y * sin_wt
    y_rot = -x * sin_wt + y * cos_wt
    return x_rot, y_rot

def dYdt_2D_moving_moon(t, Y):
    r3 = Y[0:2]
    v3 = Y[2:4]
    r2 = get_r2_moving(t)
    r3_mag = np.linalg.norm(r3)
    a1 = -G * m1 * r3 / (r3_mag**3) if r3_mag > 1e-6 else np.zeros(2)
    r23_vec = r2 - r3
    r23_mag = np.linalg.norm(r23_vec)
    a2 = G * m2 * r23_vec / (r23_mag**3) if r23_mag > 1e-6 else np.zeros(2)
    a3 = a1 + a2
    return np.concatenate((v3, a3))

# ================== НАЧАЛЬНЫЕ УСЛОВИЯ ==================
t0 = 0.0
alpha = alpha = np.pi / 4
	
V_start = 11095.0

x3_0 = R_EARTH * np.cos(alpha)
y3_0 = R_EARTH * np.sin(alpha)
vx3_0 = V_start * np.cos(alpha)
vy3_0 = V_start * np.sin(alpha)
Y0 = np.array([x3_0, y3_0, vx3_0, vy3_0])

h = 100.0                    
total_time = 864000 * 10     
total_steps = int(total_time / h)
num_initial_points = 4

# ================== ИНТЕГРАТОРЫ ==================
def runge_kutta_4th_order_vec(f, t0, Y0, h, num_points):
    t_values = [t0]
    Y_values = [Y0]
    t = t0
    Y = Y0
    for _ in range(num_points - 1):
        k1 = h * f(t, Y)
        k2 = h * f(t + h/2, Y + k1/2)
        k3 = h * f(t + h/2, Y + k2/2)
        k4 = h * f(t + h, Y + k3)
        Y_next = Y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_next = t + h
        t_values.append(t_next)
        Y_values.append(Y_next)
        t, Y = t_next, Y_next
    return np.array(t_values), np.array(Y_values)

def adams_bashforth_4th_order_vec(f, t_init, Y_init, h, total_steps):
    if len(t_init) < 4 or len(Y_init) < 4:
        raise ValueError("Адамс-Башфорт 4-го порядка требует 4 начальные точки.")
    t_values = list(t_init)
    Y_values = list(Y_init)
    F_values = [f(t, Y) for t, Y in zip(t_init, Y_init)]
    for n in range(3, total_steps):
        Y_next = Y_values[n] + (h / 24) * (
            55 * F_values[n] - 59 * F_values[n-1] + 37 * F_values[n-2] - 9 * F_values[n-3]
        )
        t_next = t_values[n] + h
        t_values.append(t_next)
        Y_values.append(Y_next)
        F_values.append(f(t_next, Y_next))
    return np.array(t_values), np.array(Y_values)

# ================== СИМУЛЯЦИЯ ==================
dYdt_2D = dYdt_2D_moving_moon
t_init, Y_init = runge_kutta_4th_order_vec(dYdt_2D, t0, Y0, h, num_initial_points)
t_sol, Y_sol = adams_bashforth_4th_order_vec(dYdt_2D, t_init, Y_init, h, total_steps)

# ================== ПОДГОТОВКА ТРАЕКТОРИИ ==================
if USE_ROTATING_FRAME:
    x_rot = []
    y_rot = []
    for i in range(len(t_sol)):
        xr, yr = to_rotating_frame(Y_sol[i, 0], Y_sol[i, 1], t_sol[i])
        x_rot.append(xr)
        y_rot.append(yr)
    x3_traj = np.array(x_rot)
    y3_traj = np.array(y_rot)
    # Вращающаяся система — Луна неподвижна
    USE_MOVING_MOON_FOR_ANIM = False
else:
    x3_traj = Y_sol[:, 0]
    y3_traj = Y_sol[:, 1]
    USE_MOVING_MOON_FOR_ANIM = True

# ================== АНИМАЦИЯ ==================
def animate_solution(x3_traj, y3_traj, t_sol, use_moving_moon):
    fig, ax = plt.subplots(figsize=(8, 8))
    VISUAL_RANGE = 1.5 * R_ORBIT
    ax.set_xlim(-VISUAL_RANGE, VISUAL_RANGE)
    ax.set_ylim(-VISUAL_RANGE, VISUAL_RANGE)
    ax.set_aspect('equal', adjustable='box')
    title = "Задача трёх тел: Луна " + ("движется" if use_moving_moon else "неподвижна")
    ax.set_title(title)
    ax.set_xlabel("X (метры)")
    ax.set_ylabel("Y (метры)")
    ax.grid(True)
    
    ax.plot(0, 0, 'o', color='blue', markersize=10, label='Земля (m1)')
    
    if use_moving_moon:
        moon_point, = ax.plot([], [], 'o', color='gray', markersize=6, label='Луна (m2)')
    else:
        ax.plot(R_ORBIT, 0, 'o', color='gray', markersize=6, label='Луна (m2)')
        moon_point = None
    
    sat_point, = ax.plot([], [], 'o', color='red', markersize=4, label='Спутник')
    sat_line, = ax.plot([], [], 'r--', linewidth=0.5, alpha=0.7)
    radius_vector_line, = ax.plot([], [], ':', color='red', linewidth=0.5)
    ax.legend(loc='upper right')
    
    def init():
        out = [sat_point, sat_line, radius_vector_line]
        if moon_point is not None:
            moon_point.set_data([], [])
            out.insert(0, moon_point)
        return out
    
    def update(i):
        out = []
        if use_moving_moon:
            r2_current = get_r2_moving(t_sol[i])
            moon_point.set_data([r2_current[0]], [r2_current[1]])
            out.append(moon_point)
        x3_current = x3_traj[i]
        y3_current = y3_traj[i]
        sat_point.set_data([x3_current], [y3_current])
        sat_line.set_data(x3_traj[:i+1], y3_traj[:i+1])
        radius_vector_line.set_data([0, x3_current], [0, y3_current])
        out.extend([sat_point, sat_line, radius_vector_line])
        return out
    
    num_frames = len(t_sol)
    step = max(1, num_frames // 2000)
    frames = range(0, num_frames, step)
    
    ani = animation.FuncAnimation(
        fig, update, frames=frames,
        init_func=init, blit=True, repeat=False, interval=20
    )
    plt.show()

# Запуск анимации
animate_solution(x3_traj, y3_traj, t_sol, use_moving_moon=USE_MOVING_MOON_FOR_ANIM)