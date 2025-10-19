import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


G = 6.67430e-11 
m1 = 5.9742e24 
m2 = 7.36e22  
R_ORBIT = 384400000

T_LUNAR_PERIOD = 27.32 * 86400.0  
OMEGA_MOON = 2 * np.pi / T_LUNAR_PERIOD 

R_ORBIT = 384400000

def get_r2(t):
    angle = OMEGA_MOON * t

    x2 = R_ORBIT * np.cos(angle)
    y2 = R_ORBIT * np.sin(angle)
    return np.array([x2, y2])

def dYdt_2D(t, Y):
    r3 = Y[0:2]
    v3 = Y[2:4]
    r2 = get_r2(t)
    
    r3_vec = r3
    r3_mag = np.linalg.norm(r3_vec)
    
    r23_vec = r2 - r3
    r23_mag = np.linalg.norm(r23_vec)
    
    a1 = -G * m1 * r3_vec / (r3_mag**3)
    
    a2 = G * m2 * r23_vec / (r23_mag**3)
    
    a3 = a1 + a2
    
    dY = np.concatenate((v3, a3))
    
    return dY


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
        t = t_next
        Y = Y_next
        
    return np.array(t_values), np.array(Y_values)

def adams_bashforth_4th_order_vec(f, t_init, Y_init, h, total_steps):
    if len(t_init) < 4 or len(Y_init) < 4:
        raise ValueError("Адамс-Башфорт 4-го порядка требует 4 начальные точки.")

    t_values = list(t_init)
    Y_values = list(Y_init)
    
    F_values = [f(t, Y) for t, Y in zip(t_init, Y_init)]
    
    for n in range(3, total_steps):
        
        F_n_minus_3 = F_values[n - 3]
        F_n_minus_2 = F_values[n - 2]
        F_n_minus_1 = F_values[n - 1]
        F_n           = F_values[n]
        
        Y_n = Y_values[n] 
        
        Y_next = Y_n + (h / 24) * (
            55 * F_n - 
            59 * F_n_minus_1 + 
            37 * F_n_minus_2 - 
            9 * F_n_minus_3
        )
        
        t_next = t_values[n] + h
        
        t_values.append(t_next)
        Y_values.append(Y_next)
        F_values.append(f(t_next, Y_next))
        
    return np.array(t_values), np.array(Y_values)


t0 = 0.0 
alpha = np.pi / 8
V_start = 11230.0

x3_0 = 6371000 * np.cos(alpha)
y3_0 = 6371000 * np.sin(alpha)

vx3_0 = V_start * np.cos(alpha)
vy3_0 = V_start * np.sin(alpha)

Y0 = np.array([x3_0, y3_0, vx3_0, vy3_0])

h = 100.0
total_time = 86400 * 3
total_steps = int(total_time / h)

num_initial_points = 4 
print("Шаг 1/3: Вычисление первых 4 точек методом РК-4...")
t_init, Y_init = runge_kutta_4th_order_vec(dYdt_2D, t0, Y0, h, num_initial_points)

print("Шаг 2/3: Вычисление траектории методом Адамса-Башфорта 4-го порядка...")
t_sol, Y_sol = adams_bashforth_4th_order_vec(dYdt_2D, t_init, Y_init, h, total_steps)

print(f"Симуляция завершена. Всего шагов: {len(Y_sol)}.")

x3_traj = Y_sol[:, 0]
y3_traj = Y_sol[:, 1]


def animate_solution(x3_traj, y3_traj, t_sol):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    VISUAL_RANGE = 1.5 * R_ORBIT

    ax.set_xlim(-VISUAL_RANGE, VISUAL_RANGE)
    ax.set_ylim(-VISUAL_RANGE, VISUAL_RANGE)

    ax.set_aspect('equal', adjustable='box') 
    ax.set_title("Задача трёх тел (Земля-Луна-Спутник) методом АБ-4")
    ax.set_xlabel("X (метры)")
    ax.set_ylabel("Y (метры)")
    ax.grid(True)
    
    earth_point, = ax.plot(0, 0, 'o', color='blue', markersize=10, label='Земля (m1)')
    
    moon_point, = ax.plot([], [], 'o', color='gray', markersize=6, label='Луна (m2)')
    
    sat_point, = ax.plot([], [], 'o', color='red', markersize=4, label='Спутник (m3)')
    
    sat_line, = ax.plot([], [], 'r--', linewidth=0.5, alpha=0.7, label='Траектория m3')

    radius_vector_line, = ax.plot([], [], ':', color='red', linewidth=0.5)
    
    ax.legend(loc='upper right')

    def init():
        moon_point.set_data([], [])
        sat_point.set_data([], [])
        sat_line.set_data([], [])
        radius_vector_line.set_data([], [])
        return moon_point, sat_point, sat_line, radius_vector_line

    def update(i):
        t_current = t_sol[i]
        
        r2_current = get_r2(t_current)
        moon_point.set_data([r2_current[0]], [r2_current[1]]) 
        
        x3_current = x3_traj[i]
        y3_current = y3_traj[i]
        
        sat_point.set_data([x3_current], [y3_current]) 
        
        sat_line.set_data(x3_traj[:i+1], y3_traj[:i+1])
        
        radius_vector_line.set_data([0, x3_current], [0, y3_current])
        
        return moon_point, sat_point, sat_line, radius_vector_line

    ani = animation.FuncAnimation(
        fig, update, frames=1000000, interval=4, blit=True, repeat=False
    )

    plt.show()

animate_solution(x3_traj, y3_traj, t_sol)
