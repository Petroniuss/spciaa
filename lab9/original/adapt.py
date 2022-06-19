import fenics as fn
from ufl import dx, ds, dot, grad
import mshr
import matplotlib.pyplot as plt

# Define L-shape domain
full_box = mshr.Rectangle(fn.Point(-1, -1), fn.Point(1, 1))
corner = mshr.Rectangle(fn.Point(-1, -1), fn.Point(0, 0))
domain = full_box - corner

# Generate initial mesh
n = 1
mesh = mshr.generate_mesh(domain, n)

# Single step of adaptive mesh refinement
cells_to_refine = fn.MeshFunction('bool', mesh, dim=2)

# mark cells that need refininig
for c in fn.cells(mesh):
    # c is an object of class dolfin.cpp.mesh.Cell
    # Example adaptation strategy: subdivide cells with centers close enough to (0, 0)
    x = c.midpoint()
    if x.norm() < 0.5:
        cells_to_refine[c] = True

# use markers to create new mesh
mesh = fn.refine(mesh, cells_to_refine)

fn.plot(mesh, title='Adapted mesh')
plt.show()

# Create finite element space
V = fn.FunctionSpace(mesh, 'Lagrange', 1)

# Exact solution
deg = 4
r = fn.Expression('sqrt(x[0]*x[0] + x[1]*x[1])', degree=deg)
theta = fn.Expression('atan2(x[0], x[1])', degree=deg)
u_exact = fn.Expression('pow(r, 2./3) * sin(2./3*theta + pi/3)', r=r, theta=theta, degree=deg, domain=mesh)

# Neumann BC
n_hat = fn.FacetNormal(mesh)
g = dot(grad(u_exact), n_hat)

# Dirichlet BC
def boundary(x, on_boundary):
    return on_boundary and (fn.near(x[0], 0) or fn.near(x[1], 0))

bc = fn.DirichletBC(V, 0, boundary)

# Define forms
u = fn.TrialFunction(V)
v = fn.TestFunction(V)

a = dot(grad(u), grad(v)) * dx
L = g * v * ds

# Compute solution
u_h = fn.Function(V, name='u')
fn.solve(a == L, u_h, bc)

# Plot the solution
p = fn.plot(u_h, title='Solution')
plt.colorbar(p)
plt.show()

# Compute errors
errorL2 = fn.errornorm(u_exact, u_h, 'L2')
errorH1 = fn.errornorm(u_exact, u_h, 'H1')
print(f'L2: {errorL2}   H1: {errorH1}')
