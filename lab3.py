import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class Solver:
    def __init__(self, ax, ay, bx, by, c, 
    left_border_condition, left_a, left_b, 
    right_border_condition, right_a, right_b, 
    bottom_border_condition, bottom_a, bottom_b, 
    top_border_condition, top_a, top_b, 
    left_border, right_border, bottom_border, top_border,
    nx, ny) -> None:

        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by
        self.c = c
        self.left_border = left_border
        self.right_border = right_border
        self.top_border = top_border
        self.bottom_border = bottom_border

        self.lx = right_border-left_border
        self.ly = top_border-bottom_border

        self.nx = nx
        self.ny = ny

        self.hx = self.lx/(nx-1)
        self.hy = self.ly/(ny-1)

        self.left_border_condition = left_border_condition
        self.right_border_condition = right_border_condition
        self.top_border_condition = top_border_condition
        self.bottom_border_condition = bottom_border_condition

        self.left_a = left_a
        self.left_b = left_b

        self.right_a = right_a
        self.right_b = right_b

        self.top_a = top_a
        self.top_b = top_b

        self.bottom_a = bottom_a
        self.bottom_b = bottom_b

    def solve_libman(self, e):
        left_border = self.left_border
        bottom_border = self.bottom_border

        hx = self.hx
        hy = self.hy

        nx = self.nx
        ny = self.ny

        left_border_condition = self.left_border_condition
        right_border_condition = self.right_border_condition
        bottom_border_condition = self.bottom_border_condition
        top_border_condition = self.top_border_condition

        left_a = self.left_a
        left_b = self.left_b

        right_a = self.right_a
        right_b = self.right_b

        top_a = self.top_a
        top_b = self.top_b

        bottom_a = self.bottom_a
        bottom_b = self.bottom_b

        hist = np.zeros((nx, ny, 0))
        u = np.zeros((nx, ny, 1))
        next_u = np.empty((nx, ny, 1))
        cur_e = np.Infinity
        while True:
            cur_e = -np.Infinity
            for x in range(1,nx-1):
                for y in range(1, ny-1):
                    next_u[x,y,0] = ((u[x-1,y,0]+u[x+1,y,0])/(hx*hx)+(u[x,y-1,0]+u[x,y+1,0])/(hy*hy))/(2*(1.0/(hx*hx)+1.0/(hy*hy)))
            for x in range(1, nx-1):
                next_u[x,0,0] = (bottom_border_condition(left_border+x*hx)-(bottom_a/hy)*next_u[x,1,0])/(bottom_b-(bottom_a/hy))
                next_u[x,-1,0] = (top_border_condition(left_border+x*hx)+(top_a/hy)*next_u[x,-2,0])/(top_b+(top_a/hy))
            for y in range(1, ny-1):
                next_u[0,y,0] = (left_border_condition(bottom_border+y*hy)-(left_a/hx)*next_u[1,y,0])/(left_b-(left_a/hx))
                next_u[-1,y,0] = (right_border_condition(bottom_border+y*hy)+(right_a/hx)*next_u[-2,y,0])/(right_b+(right_a/hx))
            for x in range(1,nx-1):
                for y in range(1, ny-1):
                    cur_e = max(cur_e, np.abs(next_u[x,y,0]-u[x,y,0]))
            u, next_u = next_u, u
            hist = np.append(hist, u, 2)
            if not cur_e > e and cur_e != 0.0:
                break   
        return hist
    def solve_seidel(self, e):
        left_border = self.left_border
        bottom_border = self.bottom_border

        hx = self.hx
        hy = self.hy

        nx = self.nx
        ny = self.ny

        left_border_condition = self.left_border_condition
        right_border_condition = self.right_border_condition
        bottom_border_condition = self.bottom_border_condition
        top_border_condition = self.top_border_condition

        left_a = self.left_a
        left_b = self.left_b

        right_a = self.right_a
        right_b = self.right_b

        top_a = self.top_a
        top_b = self.top_b

        bottom_a = self.bottom_a
        bottom_b = self.bottom_b

        hist = np.zeros((nx, ny, 0))
        u = np.zeros((nx, ny, 1))
        cur_e = np.Infinity
        while True:
            cur_e = -np.Infinity
            for x in range(1,nx-1):
                for y in range(1, ny-1):
                    cur_e = max(cur_e, abs(u[x,y]-((u[x-1,y,0]+u[x+1,y,0])/(hx*hx)+(u[x,y-1,0]+u[x,y+1,0])/(hy*hy))/(2*(1.0/(hx*hx)+1.0/(hy*hy)))))
                    u[x,y,0] = ((u[x-1,y,0]+u[x+1,y,0])/(hx*hx)+(u[x,y-1,0]+u[x,y+1,0])/(hy*hy))/(2*(1.0/(hx*hx)+1.0/(hy*hy)))
            for x in range(1, nx-1):
                u[x,0,0] = (bottom_border_condition(left_border+x*hx)-(bottom_a/hy)*u[x,1,0])/(bottom_b-(bottom_a/hy))
                u[x,-1,0] = (top_border_condition(left_border+x*hx)+(top_a/hy)*u[x,-2,0])/(top_b+(top_a/hy))
            for y in range(1, ny-1):
                u[0,y,0] = (left_border_condition(bottom_border+y*hy)-(left_a/hx)*u[1,y,0])/(left_b-(left_a/hx))
                u[-1,y,0] = (right_border_condition(bottom_border+y*hy)+(right_a/hx)*u[-2,y,0])/(right_b+(right_a/hx))
            hist = np.append(hist, u, 2)
            print(cur_e)
            if not cur_e > e and cur_e != 0.0:
                break   
        return hist
    def solve_libman_relaxed(self, e, w = 1):
        left_border = self.left_border
        bottom_border = self.bottom_border

        hx = self.hx
        hy = self.hy

        nx = self.nx
        ny = self.ny

        left_border_condition = self.left_border_condition
        right_border_condition = self.right_border_condition
        bottom_border_condition = self.bottom_border_condition
        top_border_condition = self.top_border_condition

        left_a = self.left_a
        left_b = self.left_b

        right_a = self.right_a
        right_b = self.right_b

        top_a = self.top_a
        top_b = self.top_b

        bottom_a = self.bottom_a
        bottom_b = self.bottom_b

        hist = np.zeros((nx, ny, 0))
        u = np.zeros((nx, ny, 1))
        next_u = np.empty((nx, ny, 1))
        cur_e = np.Infinity
        while True:
            cur_e = -np.Infinity
            for x in range(1,nx-1):
                for y in range(1, ny-1):
                    next = ((u[x-1,y,0]+u[x+1,y,0])/(hx*hx)+(u[x,y-1,0]+u[x,y+1,0])/(hy*hy))/(2*(1.0/(hx*hx)+1.0/(hy*hy)))
                    next_u[x,y,0] = next+w*(next-u[x,y])
            for x in range(1, nx-1):
                next_u[x,0,0] = (bottom_border_condition(left_border+x*hx)-(bottom_a/hy)*next_u[x,1,0])/(bottom_b-(bottom_a/hy))
                next_u[x,-1,0] = (top_border_condition(left_border+x*hx)+(top_a/hy)*next_u[x,-2,0])/(top_b+(top_a/hy))
            for y in range(1, ny-1):
                next_u[0,y,0] = (left_border_condition(bottom_border+y*hy)-(left_a/hx)*next_u[1,y,0])/(left_b-(left_a/hx))
                next_u[-1,y,0] = (right_border_condition(bottom_border+y*hy)+(right_a/hx)*next_u[-2,y,0])/(right_b+(right_a/hx))
            for x in range(1,nx-1):
                for y in range(1, ny-1):
                    cur_e = max(cur_e, np.abs(next_u[x,y,0]-u[x,y,0]))
            u, next_u = next_u, u
            hist = np.append(hist, u, 2)
            if not cur_e > e and cur_e != 0.0:
                break   
        return hist

# вариант 3
def true_u(x, y):
    return np.exp(x)*np.cos(y)

left_border = 0
right_border = 1
bottom_border = 0
top_border = np.pi/2

def left_border_condition(y):
    return np.cos(y)
def right_border_condition(y):
    return np.e*np.cos(y)
def bottom_border_condition(x):
    return 0.0
def top_border_condition(x):
    return -np.exp(x)

nx = 10
ny = 10

solver = Solver(1, 1, 0, 0, 0,
left_border_condition, 0, 1,
right_border_condition, 0, 1,
bottom_border_condition, 1, 0,
top_border_condition, 1, 0,
left_border, right_border, bottom_border, top_border,
nx, ny)

tu = np.empty((solver.nx, solver.ny))
for x in range(solver.nx):
    for y in range(solver.ny):
        tu[x,y] = true_u(left_border+(x)*solver.hx, bottom_border+(y)*solver.hy)

u_libman = solver.solve_libman(0.0001)
print('u_libman')
u_seidel = solver.solve_seidel(0.0001)
print('u_seidel')
u_libman_relaxed = solver.solve_libman_relaxed(0.0001, 1.001)
print('u_libman_relaxed')

y = np.linspace(left_border, right_border, solver.nx)
x = np.linspace(bottom_border, top_border, solver.ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
# plots
u_libman_ax = fig.add_subplot(2, 3, 1, projection='3d')
Z = u_libman[:, :, 0]
u_libman_plot = u_libman_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
u_seidel_ax = fig.add_subplot(2, 3, 2, projection='3d')
Z = u_seidel[:, :, 0]
u_seidel_plot = u_seidel_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
u_libman_relaxed_ax = fig.add_subplot(2, 3, 3, projection='3d')
Z = u_libman_relaxed[:, :, 0]
u_libman_relaxed_plot = u_libman_relaxed_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
# errors
u_libman_error_ax = fig.add_subplot(2, 3, 4, projection='3d')
Z = abs(u_libman[:,:,0]-tu[:,:])
u_libman_error = u_libman_error_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
u_seidel_error_ax = fig.add_subplot(2, 3, 5, projection='3d')
Z = abs(u_seidel[:,:,0]-tu[:,:])
u_seidel_error = u_seidel_error_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
u_libman_relaxed_error_ax = fig.add_subplot(2, 3, 6, projection='3d')
Z = abs(u_libman_relaxed[:,:,0]-tu[:,:])
u_libman_relaxed_error = u_libman_relaxed_error_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')


# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.1, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.1, 0.1, 0.8, 0.03])
time_slider = Slider(
    ax=axfreq,
    label='Time',
    valmin = 0,
    valmax=max(u_libman.shape[2], u_seidel.shape[2], u_libman_relaxed.shape[2]),
    valinit = 0,
)

print(u_libman.shape[2], u_seidel.shape[2], u_libman_relaxed.shape[2])

# The function to be called anytime a slider's value changes
def update(val):
    u_libman_ax.clear()
    u_libman_ax.plot_surface(X, Y, u_libman[:, :, min(u_libman.shape[2]-1, int(val))], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    u_seidel_ax.clear()
    u_seidel_ax.plot_surface(X, Y, u_seidel[:, :, min(u_seidel.shape[2]-1, int(val))], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    u_libman_relaxed_ax.clear()
    u_libman_relaxed_ax.plot_surface(X, Y, u_libman_relaxed[:, :, min(u_libman_relaxed.shape[2]-1, int(val))], rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    u_libman_error_ax.clear()
    u_libman_error_ax.plot_surface(X, Y, abs(u_libman[:, :, min(u_libman.shape[2]-1, int(val))]-tu[:,:]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    u_seidel_error_ax.clear()
    u_seidel_error_ax.plot_surface(X, Y, abs(u_seidel[:, :, min(u_seidel.shape[2]-1, int(val))]-tu[:,:]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    u_libman_relaxed_error_ax.clear()
    u_libman_relaxed_error_ax.plot_surface(X, Y, abs(u_libman_relaxed[:, :, min(u_libman_relaxed.shape[2]-1, int(val))]-tu[:,:]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()