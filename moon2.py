import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ================== ПАРАМЕТРЫ ==================
G = 6.67430e-11
m1 = 5.9742e24      # Земля
m2 = 7.36e22        # Луна
R_ORBIT = 384400000 # Среднее расстояние Земля–Луна
T_LUNAR_PERIOD = 27.32 * 86400.0
OMEGA_MOON = 2 * np.pi / T_LUNAR_PERIOD
R_EARTH = 6371000

r2_fixed = np.array([R_ORBIT, 0.0])

USE_MOVING_MOON = True  # важно для анимации

def get_r2_moving(t):
    angle = OMEGA_MOON * t
    return np.array([R_ORBIT * np.cos(angle), R_ORBIT * np.sin(angle)])

# ================== УРАВНЕНИЯ ДВИЖЕНИЯ ==================
def dYdt_2D_moving_moon_with_maneuvers(t, Y, maneuvers=None):
    r3 = Y[0:2]
    v3 = Y[2:4]
    r2 = get_r2_moving(t)
    
    # Гравитация от Земли
    r3_mag = np.linalg.norm(r3)
    a1 = -G * m1 * r3 / (r3_mag**3) if r3_mag > 1e-6 else np.zeros(2)
    
    # Гравитация от Луны
    r23_vec = r2 - r3
    r23_mag = np.linalg.norm(r23_vec)
    a2 = G * m2 * r23_vec / (r23_mag**3) if r23_mag > 1e-6 else np.zeros(2)
    
    a_total = a1 + a2

    # Учёт тяги, если есть манёвры
    if maneuvers is not None:
        for man in maneuvers:
            t0 = man["start_time"]
            if t0 <= t <= t0 + 1800:  # манёвр длится 30 минут = 1800 с
                dir_vec = man["direction"]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-8:
                    unit_dir = dir_vec / dir_norm
                    a_thrust = (500.0 / 1800.0) * unit_dir  # ≈0.2778 м/с²
                    a_total += a_thrust
                break  # предполагаем, что манёвры не перекрываются

    return np.concatenate((v3, a_total))

def dYdt_2D_fixed_moon(t, Y):
    r3 = Y[0:2]
    v3 = Y[2:4]
    r2 = r2_fixed  
    r3_mag = np.linalg.norm(r3)
    a1 = -G * m1 * r3 / (r3_mag**3) if r3_mag > 1e-6 else np.zeros(2)
    r23_vec = r2 - r3
    r23_mag = np.linalg.norm(r23_vec)
    a2 = G * m2 * r23_vec / (r23_mag**3) if r23_mag > 1e-6 else np.zeros(2)
    a3 = a1 + a2
    return np.concatenate((v3, a3))

# ================== НАЧАЛЬНЫЕ УСЛОВИЯ ==================
t0 = 0.0
alpha = np.pi * 95 / 400
V_start = 11100

x3_0 = R_EARTH * np.cos(alpha)
y3_0 = R_EARTH * np.sin(alpha)
vx3_0 = V_start * np.cos(alpha)
vy3_0 = V_start * np.sin(alpha)
Y0 = np.array([x3_0, y3_0, vx3_0, vy3_0])

h = 100.0
total_time = 864000 * 10  # ~100 дней
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
        Y_n = Y_values[n]
        Y_next = Y_n + (h / 24) * (
            55 * F_values[n] - 59 * F_values[n-1] + 37 * F_values[n-2] - 9 * F_values[n-3]
        )
        t_next = t_values[n] + h
        t_values.append(t_next)
        Y_values.append(Y_next)
        F_values.append(f(t_next, Y_next))
    return np.array(t_values), np.array(Y_values)

# ================== ПЕРВАЯ СИМУЛЯЦИЯ: БЕЗ МАНЁВРОВ ==================
print("Выполняется первая симуляция (без манёвров)...")
dYdt_base = lambda t, Y: dYdt_2D_moving_moon_with_maneuvers(t, Y, maneuvers=None)

t_init, Y_init = runge_kutta_4th_order_vec(dYdt_base, t0, Y0, h, num_initial_points)
t_sol, Y_sol = adams_bashforth_4th_order_vec(dYdt_base, t_init, Y_init, h, total_steps)

# ================== АНАЛИЗ ТРАЕКТОРИИ ==================
r_moon_traj = np.array([get_r2_moving(t) for t in t_sol])
dist_to_moon = np.linalg.norm(Y_sol[:, :2] - r_moon_traj, axis=1)
idx_min = np.argmin(dist_to_moon)
t_closest = t_sol[idx_min]
dist_min = dist_to_moon[idx_min]

print(f"Минимальное расстояние до Луны: {dist_min/1000:.1f} км в момент t = {t_closest/3600:.2f} ч")

# ================== ПЛАНИРОВАНИЕ МАНЁВРА ==================
if dist_min < 5e7:  # если ближе 50 000 км — можно тормозить
    t_maneuver_start = max(0.0, t_closest - 600)  # начать за 10 мин до сближения
    idx_maneuver = np.argmin(np.abs(t_sol - t_maneuver_start))
    r3_at_man = Y_sol[idx_maneuver, :2]
    v3_at_man = Y_sol[idx_maneuver, 2:4]
    r2_at_man = get_r2_moving(t_maneuver_start)
    v2_at_man = np.array([-OMEGA_MOON * r2_at_man[1], OMEGA_MOON * r2_at_man[0]])
    v_rel = v3_at_man - v2_at_man
    direction = -v_rel / (np.linalg.norm(v_rel) + 1e-12)

    maneuvers = [{"start_time": t_maneuver_start, "direction": direction}]
    print(f"→ Планируется манёвр торможения в t = {t_maneuver_start/3600:.2f} ч")
else:
    maneuvers = None
    print("→ Манёвр не планируется: аппарат не подлетает достаточно близко.")

# ================== ВТОРАЯ СИМУЛЯЦИЯ (С МАНЁВРОМ) ==================
if maneuvers is not None:
    print("Выполняется вторая симуляция (с манёвром)...")
    dYdt_with_man = lambda t, Y: dYdt_2D_moving_moon_with_maneuvers(t, Y, maneuvers=maneuvers)
    t_init2, Y_init2 = runge_kutta_4th_order_vec(dYdt_with_man, t0, Y0, h, num_initial_points)
    t_sol2, Y_sol2 = adams_bashforth_4th_order_vec(dYdt_with_man, t_init2, Y_init2, h, total_steps)
    x3_traj, y3_traj, t_sol_anim = Y_sol2[:, 0], Y_sol2[:, 1], t_sol2
else:
    x3_traj, y3_traj, t_sol_anim = Y_sol[:, 0], Y_sol[:, 1], t_sol

# ================== АНИМАЦИЯ ==================
def animate_solution(x3_traj, y3_traj, t_sol):
    fig, ax = plt.subplots(figsize=(8, 8))
    VISUAL_RANGE = 1.5 * R_ORBIT
    ax.set_xlim(-VISUAL_RANGE, VISUAL_RANGE)
    ax.set_ylim(-VISUAL_RANGE, VISUAL_RANGE)
    ax.set_aspect('equal', adjustable='box')
    title = "Задача трёх тел: Луна " + ("движется" if USE_MOVING_MOON else "неподвижна")
    ax.set_title(title)
    ax.set_xlabel("X (метры)")
    ax.set_ylabel("Y (метры)")
    ax.grid(True)
    
    ax.plot(0, 0, 'o', color='blue', markersize=10, label='Земля (m1)')
    
    if USE_MOVING_MOON:
        moon_point, = ax.plot([], [], 'o', color='gray', markersize=6, label='Луна (m2)')
    else:
        ax.plot(r2_fixed[0], r2_fixed[1], 'o', color='gray', markersize=6, label='Луна (m2)')
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
        if USE_MOVING_MOON:
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
animate_solution(x3_traj, y3_traj, t_sol_anim)