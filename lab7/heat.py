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
u0 = fn.Expression('0', degree=2)
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
F = du_dt * v + dot(grad(u), grad(v)) - f * v

a, L = fn.system(F * dx)  # FEniCS can separate F into left- and right-hand side
                          # "* dx" denotes integration over the domain

# Output file for ParaView
out = fn.XDMFFile('heat.xdmf')
out.parameters['flush_output'] = True              # to see partial results during simulation
out.parameters['rewrite_function_mesh'] = False    # mesh does not change between time steps
out.write(u_h, t=0)


# Time stepping loop
for n in range(1, num_steps+1):
    t = n * dt
    print('Step {}, t = {:.2f}'.format(n, t))

    # NOT u_prev = u_h - that would not modify u_prev in form definition
    u_prev.assign(u_h)

    # Solve equation for the current time step
    fn.solve(a == L, u_h, bc)
    out.write(u_h, t)

