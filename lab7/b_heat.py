from cProfile import label
import fenics as fn
import mshr
from ufl import grad, dot, dx, ds
import numpy as np
import matplotlib.pyplot as plt

# Define mesh
N = 32
domain = mshr.Circle(fn.Point(0, 0), 2)
mesh = mshr.generate_mesh(domain, N)

# Time stepping parameters
T = 1.0
num_steps = 50
dt = T / num_steps

# Create finite element space
V = fn.FunctionSpace(mesh, 'Lagrange', 1)
u_h = fn.Function(V, name='heat')    # buffer for current solution

# Initial state
# -- updated initial state.
u0 = fn.Expression('pow(e, -5*(((x[0]+0.5)*(x[0]+0.5))+((x[1]+0.5)*(x[1]+0.5))))', e=np.e, degree=2)
u_h.interpolate(u0)

# Forcing
f = fn.Constant(0)

# Uniform Dirichlet boundary conditions
bc = fn.DirichletBC(V, 0, 'on_boundary')

# Define forms
u = fn.TrialFunction(V)
v = fn.TestFunction(V)
u_prev = fn.Function(V)   # solution from the previous time step
k = fn.Constant(dt)       # not strictly necessary, but makes it easier for FEniCS
                          # to generate efficient code

# Define the problem to solve in each time step: F(u, u_prev, v) = 0
du_dt = (u - u_prev) / k
# -- Crank-Nicolson instead of implicit Euler.
F = du_dt * v + .5*(dot(grad(u), grad(v)) + dot(grad(u_prev), grad(v))) - f * v

a, L = fn.system(F * dx)  # FEniCS can separate F into left- and right-hand side
                          # "* dx" denotes integration over the domain

# Output file for ParaView
out = fn.XDMFFile('heat.xdmf')
out.parameters['flush_output'] = True              # to see partial results during simulation
out.parameters['rewrite_function_mesh'] = False    # mesh does not change between time steps
out.write(u_h, t=0)

q_y = []
phi_y = []


# Time stepping loop
for n in range(1, num_steps+1):
    t = n * dt
    print('Step {}, t = {:.2f}'.format(n, t))

    ## -- compute Q and Phi.
    Q_form = u_prev * dx
    Q = fn.assemble(Q_form)

    n = fn.FacetNormal(mesh)
    Phi_form = dot(grad(u_prev), n) * ds
    Phi = fn.assemble(Phi_form)

    print(f"Q = {Q}, Fi = {Phi}")

    q_y.append(Q)
    phi_y.append(Phi)

    # NOT u_prev = u_h - that would not modify u_prev in form definition
    u_prev.assign(u_h)

    # Solve equation for the current time step
    fn.solve(a == L, u_h, bc)
    # -- draw a solution at each timestep
    out.write(u_h, t)

plt.plot(range(len(q_y)), q_y, label='Q')
plt.xlabel('time step')
plt.ylabel('Value')

plt.plot(range(len(phi_y)), phi_y, label='Fi')
plt.legend()
plt.grid()
plt.savefig(f'Q_Fi_{num_steps}.png')
