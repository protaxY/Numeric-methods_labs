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
    nx, ny,
    end_time,
    time_steps,
    u_start) -> None:

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

        self.end_time = end_time
        self.time_steps = time_steps
        self.tau = end_time/time_steps

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

        self.u_start = u_start

    def solve_variable_direction_method(self):
        ax = self.ax
        ay = self.ay
        bx = self.bx
        by = self.by
        c = self.c
        left_border = self.left_border
        right_border = self.right_border
        top_border = self.top_border
        bottom_border = self.bottom_border

        lx = self.right_border-left_border
        ly = self.top_border-bottom_border

        nx = self.nx
        ny = self.ny

        hx = self.hx
        hy = self.hy

        end_time = self.end_time
        time_steps = self.time_steps
        tau = self.tau

        left_border_condition = self.left_border_condition
        right_border_condition = self.right_border_condition
        top_border_condition = self.top_border_condition
        bottom_border_condition = self.bottom_border_condition

        left_a = self.left_a
        left_b = self.left_b

        right_a = self.right_a
        right_b = self.right_b

        top_a = self.top_a
        top_b = self.top_b

        bottom_a = self.bottom_a
        bottom_b = self.bottom_b

        u_start = self.u_start

        hist = np.zeros((ny, nx, 0))

        start = np.empty((ny, nx, 1))
        for y in range(ny):
            for x in range(nx):
                start[y, x] = u_start(left_border+hx*x, bottom_border+hy*y)

        hist = np.append(hist, start, 2)

        for k in range(0, time_steps):
            half_u = np.zeros((ny, nx, 1))
            next_u = np.zeros((ny, nx, 1))

            for y in range(1, ny-1):
                A = np.zeros((nx, nx))
                d = np.empty(nx)

                A[0, 0] = (-(left_a/hx)+left_b)
                A[0, 1] = (left_a/hx)
                A[-1, -1] = ((right_a/hx)+right_b)
                A[-1, -2] = -(right_a/hx)

                d[0] = left_border_condition(bottom_border+hy*y, tau*k+(tau/2))
                d[-1] = right_border_condition(bottom_border+hy*y, tau*k+(tau/2))

                for x in range(1, nx-1):
                    A[x, x-1] = -(ax/(hx**2))+bx/(2*hx)
                    A[x, x] = (2/tau)+((2*ax)/(hx**2))-c
                    A[x, x+1] = -(ax/(hx**2))-bx/(2*hx)
                    d[x] = (1/(tau/2))*hist[y, x, -1]+(ay/hy**2)*(hist[y+1, x, -1]-2*hist[y, x, -1]+hist[y-1, x, -1])+(by/(2*hy))*(hist[y+1, x, -1]-hist[y-1, x, -1])

                solution = np.linalg.solve(A, d)
                for x in range(nx):
                    half_u[y, x, 0] = solution[x]

            for x in range(nx):
                half_u[0, x, -1] = (bottom_border_condition(left_border+hx*x, tau*k+(tau/2))-(bottom_a/hy)*half_u[1, x, -1])/((-bottom_a/hy)+bottom_b)
                half_u[-1, x, -1] = (top_border_condition(left_border+hx*x, tau*k+(tau/2))+(top_a/hy)*half_u[-2, x, -1])/((top_a/hy)+top_b)

            for x in range(1, nx-1):
                A = np.zeros((ny, ny))
                d = np.empty(ny)

                A[0, 0] = (-(bottom_a/hy)+bottom_b)
                A[0, 1] = (bottom_a/hy)
                A[-1, -1] = ((top_a/hy)+top_b)
                A[-1, -2] = -(top_a/hy)

                d[0] = bottom_border_condition(left_border+hx*x, tau*k+(tau/2))
                d[-1] = top_border_condition(left_border+hx*x, tau*k+(tau/2))

                for y in range(1, ny-1):
                    A[y, y-1] = -(ay/(hy**2))+by/(2*hy)
                    A[y, y] = (2/tau)+((2*ay)/(hy**2))-c
                    A[y, y+1] = -(ay/(hy**2))-by/(2*hy)
                    d[y] = (1/(tau/2))*half_u[y, x, -1]+(ax/hx**2)*(half_u[y, x+1, -1]-2*half_u[y, x, -1]+half_u[y, x-1, -1])+(bx/(2*hx))*(half_u[y, x+1, -1]-half_u[y, x-1, -1])
                
                solution = np.linalg.solve(A, d)
                for y in range(ny):
                    next_u[y, x, 0] = solution[y]

            for y in range(ny):
                next_u[y, 0, -1] = (left_border_condition(bottom_border+hy*y, tau*k+(tau/2))-(left_a/hx)*next_u[y, 1, -1])/((-left_a/hx)+left_b)
                next_u[y, -1, -1] = (right_border_condition(bottom_border+hy*y, tau*k+(tau/2))+(right_a/hx)*next_u[y, -2, -1])/((right_a/hx)+right_b)

            hist = np.append(hist, next_u, 2)
        return hist

    def solve_fractional_step_method(self):
        ax = self.ax
        ay = self.ay
        bx = self.bx
        by = self.by
        c = self.c
        left_border = self.left_border
        right_border = self.right_border
        top_border = self.top_border
        bottom_border = self.bottom_border

        lx = self.right_border-left_border
        ly = self.top_border-bottom_border

        nx = self.nx
        ny = self.ny

        hx = self.hx
        hy = self.hy

        end_time = self.end_time
        time_steps = self.time_steps
        tau = self.tau

        left_border_condition = self.left_border_condition
        right_border_condition = self.right_border_condition
        top_border_condition = self.top_border_condition
        bottom_border_condition = self.bottom_border_condition

        left_a = self.left_a
        left_b = self.left_b

        right_a = self.right_a
        right_b = self.right_b

        top_a = self.top_a
        top_b = self.top_b

        bottom_a = self.bottom_a
        bottom_b = self.bottom_b

        u_start = self.u_start

        hist = np.zeros((ny, nx, 0))

        start = np.empty((ny, nx, 1))
        for y in range(ny):
            for x in range(nx):
                start[y, x] = u_start(left_border+hx*x, bottom_border+hy*y)

        hist = np.append(hist, start, 2)

        for k in range(1, time_steps+1):
            half_u = np.zeros((ny, nx, 1))
            next_u = np.zeros((ny, nx, 1))

            for y in range(1, ny-1):
                A = np.zeros((nx, nx))
                d = np.empty(nx)

                A[0, 0] = (-(left_a/hx)+left_b)
                A[0, 1] = (left_a/hx)
                A[-1, -1] = ((right_a/hx)+right_b)
                A[-1, -2] = -(right_a/hx)

                d[0] = left_border_condition(bottom_border+hy*y, tau*(k+0.5))
                d[-1] = right_border_condition(bottom_border+hy*y, tau*(k+0.5))

                for x in range(1, nx-1):
                    A[x, x-1] = (ax/(hx**2))-bx/(2*hx)
                    A[x, x] = (-1/tau)-(2*ax)/(hx**2)+c
                    A[x, x+1] = (ax/(hx**2))+bx/(2*hx)
                    d[x] = (-1/tau)*hist[y, x, -1]

                solution = np.linalg.solve(A, d)
                for x in range(nx):
                    half_u[y, x, 0] = solution[x]

            for x in range(nx):
                half_u[0, x] = (bottom_border_condition(left_border+hx*x, tau*(k+0.5))-(bottom_a/hy)*half_u[1, x])/((-bottom_a/hy)+bottom_b)
                half_u[-1, x] = (top_border_condition(left_border+hx*x, tau*(k+0.5))+(top_a/hy)*half_u[-2, x])/((top_a/hy)+top_b)

            for x in range(1, nx-1):
                A = np.zeros((ny, ny))
                d = np.empty(ny)

                A[0, 0] = (-(bottom_a/hy)+bottom_b)
                A[0, 1] = (bottom_a/hy)
                A[-1, -1] = ((top_a/hy)+top_b)
                A[-1, -2] = -(top_a/hy)

                d[0] = bottom_border_condition(left_border+hx*x, tau*(k))
                d[-1] = top_border_condition(left_border+hx*x, tau*(k))

                for y in range(1, ny-1):
                    A[y, y-1] = (ay/(hy**2))-by/(2*hy)
                    A[y, y] = (-1/tau)-(2*ay)/(hy**2)+c
                    A[y, y+1] = (ay/(hy**2))+by/(2*hy)
                    d[y] = (-1/tau)*half_u[y, x, -1]
                
                solution = np.linalg.solve(A, d)
                for y in range(ny):
                    next_u[y, x, 0] = solution[y]

            for y in range(ny):
                next_u[y, 0] = (left_border_condition(bottom_border+hy*y, tau*(k))-(left_a/hx)*next_u[y, 1])/((-left_a/hx)+left_b)
                next_u[y, -1] = (right_border_condition(bottom_border+hy*y, tau*(k))+(right_a/hx)*next_u[y, -2])/((right_a/hx)+right_b)

            hist = np.append(hist, next_u, 2)
        return hist

# вариант 4
true_a = 1
def true_u(x, y, t):
    return np.cos(2*x)*np.cosh(y)*np.exp(-3*true_a*t)

def u_start(x, y):
    return np.cos(2*x)*np.cosh(y)

left_border = 0
right_border = np.pi/4
bottom_border = 0
top_border = np.log(2)

def left_border_condition(y, t):
    return np.cosh(y)*np.exp(-3*true_a*t)
def right_border_condition(y, t):
    return 0.0
def bottom_border_condition(x, t):
    return np.cos(2*x)*np.exp(-3*true_a*t)
def top_border_condition(x, t):
    return (3/4)*np.cos(2*x)*np.exp(-3*true_a*t)

nx = 40
ny = 40

end_time = 2
time_steps = 50

solver = Solver(1, 1, 0, 0, 0,
left_border_condition, 0, 1,
right_border_condition, 0, 1,
bottom_border_condition, 0, 1,
top_border_condition, 1, 0,
left_border, right_border, bottom_border, top_border,
nx, ny,
end_time,
time_steps,
u_start)

tu = np.empty((solver.nx, solver.ny, solver.time_steps))
for k in range(0, solver.time_steps):
    for x in range(solver.nx):
        for y in range(solver.ny):
            tu[y,x,k] = true_u(left_border+(x)*solver.hx, bottom_border+(y)*solver.hy, solver.tau*k)

u_variable_direction_method = solver.solve_variable_direction_method()
u_fractional_step_method = solver.solve_fractional_step_method()

y = np.linspace(left_border, right_border, solver.nx)
x = np.linspace(bottom_border, top_border, solver.ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
# plots
u_variable_direction_method_ax = fig.add_subplot(2, 2, 1, projection='3d')
Z = u_variable_direction_method[:, :, 0]
u_variable_direction_method_plot = u_variable_direction_method_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                                                               cmap='viridis', edgecolor='none')
u_variable_direction_method_ax.set_title('Fractional step method')
u_fractional_step_method_ax = fig.add_subplot(2, 2, 2, projection='3d')
Z = u_fractional_step_method[:, :, 0]
u_fractional_step_method_plot = u_fractional_step_method_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                                                         cmap='viridis', edgecolor='none')
u_fractional_step_method_ax.set_title('Variable direction method')

# errors
u_variable_direction_method_error_ax = fig.add_subplot(2, 2, 3)
# Z = abs(u_variable_direction_method[:,:,0]-tu[:,:,0])
# u_variable_direction_method_error = u_variable_direction_method_error_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                                                                                       cmap='viridis', edgecolor='none')
errors = abs(u_variable_direction_method[:,:,:-1]-tu)
errors = np.max(errors, axis=(0, 1))
u_variable_direction_method_error_ax.plot(np.linspace(0, solver.end_time, solver.time_steps), errors)

plt.title('Error')
u_fractional_step_method_error_ax = fig.add_subplot(2, 2, 4)
# Z = abs(u_fractional_step_method[:,:,0]-tu[:,:,0])
# u_fractional_step_method_error = u_fractional_step_method_error_ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                                                                                 cmap='viridis', edgecolor='none')
errors = abs(u_fractional_step_method[:,:,:-1]-tu)
errors = np.max(errors, axis=(0, 1))
u_fractional_step_method_error_ax.plot(np.linspace(0, solver.end_time, solver.time_steps), errors)
plt.title('Error')


# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.1, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.1, 0.1, 0.8, 0.03])
time_slider = Slider(
    ax=axfreq,
    label='Time',
    valmin = 0,
    valmax=time_steps,
    valinit = 0,
)

# The function to be called anytime a slider's value changes
def update(val):
    val = int(val)

    u_variable_direction_method_ax.clear()
    u_variable_direction_method_ax.plot_surface(X, Y, u_variable_direction_method[:, :, val], rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    u_fractional_step_method_ax.clear()
    u_fractional_step_method_ax.plot_surface(X, Y, u_fractional_step_method[:, :, val], rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # u_variable_direction_method_error_ax.clear()
    # u_variable_direction_method_error_ax.plot_surface(X, Y, abs(u_variable_direction_method[:, :, val]-tu[:,:,val]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    
    # u_fractional_step_method_error_ax.clear()
    # u_fractional_step_method_error_ax.plot_surface(X, Y, abs(u_fractional_step_method[:, :, val]-tu[:,:,val]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()