import fenics as fn
from ufl import dx, ds, grad, dot
import matplotlib.pyplot as plt

# Create uniform 8x8 mesh on [0, 1] x [0, 1]
n = 8
mesh = fn.UnitSquareMesh(n, n)

# Create finite element space on the mesh (1st order polynomials)
V = fn.FunctionSpace(mesh, 'Lagrange', 1)

# Define exact solution and forcing
# u_exact = fn.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
u_exact = fn.Expression('(sin(pi * x[0]) * cos(pi * x[1])) / (2 * pi * pi)', degree=5)

# modification:
# f = fn.Constant(-6.0)
f = fn.Expression('sin(pi * x[0]) * cos(pi * x[1])', degree=2)

# Define boundary conditions - Dirichlet BC on the entire boundary
def boundary(x, on_boundary):
    return on_boundary

bc = fn.DirichletBC(V, u_exact, boundary)

# Define forms
u = fn.TrialFunction(V)
v = fn.TestFunction(V)

a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute solution
u_h = fn.Function(V, name='potential')
fn.solve(a == L, u_h, bc)

# Plot the solution
fn.plot(mesh)
fn.plot(u_h, title='Electric potential')
plt.show()

# Save to file
out = fn.XDMFFile('poisson.xdmf')
out.parameters['functions_share_mesh'] = True
out.write(u_h, t=0)

# Compute L2 error
normL2_exact = fn.norm(u_exact, 'L2', mesh)
normL2 = fn.norm(u_h, 'L2')
errorL2 = fn.errornorm(u_exact, u_h, 'L2')
print('L2 error: {:7g} ({:6.3%})'.format(errorL2, errorL2 / normL2_exact))

# Compute max error (sup norm)
vals_u_h = u_h.compute_vertex_values(mesh)
vals_u_exact = u_exact.compute_vertex_values(mesh)
error_max = abs(vals_u_h - vals_u_exact).max()
exact_max = abs(vals_u_exact).max()
print('Max error: {:7g} ({:6.3%})'.format(error_max, error_max / exact_max))

# Save electric field to file
W = fn.VectorFunctionSpace(mesh, 'Lagrange', 1)
E = fn.Function(W, name='electric field')
fn.project(- grad(u_h), W, function=E)
out.write(E, t=0)

# Post-processing
# energy of the field
energy_form = 1/2 * dot(grad(u_h), grad(u_h)) * dx
energy = fn.assemble(energy_form)
print('Energy: {}'.format(energy))

# flux through boundary
n = fn.FacetNormal(mesh)
flux_form = - dot(grad(u_h), n) * ds
flux = fn.assemble(flux_form)
print('Flux:   {} (exact is -6)'.format(flux))

