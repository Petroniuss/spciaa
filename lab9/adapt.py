from tkinter import W
import fenics as fn
from ufl import dx, ds, dot, grad
import mshr
import matplotlib.pyplot as plt

def near_origin_strategy(mesh, u_exact, u):
    cells_to_refine = fn.MeshFunction('bool', mesh, dim=2)
    for c in fn.cells(mesh):
        x = c.midpoint()
        if x.norm() < 0.5:
            cells_to_refine[c] = True
    
    return cells_to_refine


def uniform_strategy(mesh, u_exact, u):
    cells_to_refine = fn.MeshFunction('bool', mesh, dim=2)
    for c in fn.cells(mesh):
        cells_to_refine[c] = True

    return cells_to_refine


def adaptive_strategy(mesh, u_exact, u):
    cells_to_refine = fn.MeshFunction('bool', mesh, dim=2)
    E_max = 0
    Es = []
    for c in fn.cells(mesh):
        E = pow(u_exact(c.midpoint()) - u(c.midpoint()), 2) * c.volume()
        Es.append(E)
        E_max = max(E_max, E)
    for i, c in enumerate(fn.cells(mesh)):
        if Es[i] > .3 * E_max:
            cells_to_refine[c] = True

    return cells_to_refine

# Define L-shape domain
full_box = mshr.Rectangle(fn.Point(-1, -1), fn.Point(1, 1))
corner = mshr.Rectangle(fn.Point(-1, -1), fn.Point(0, 0))
domain = full_box - corner

strategies = [
    {
        'name': 'near origin',
        'strategy': near_origin_strategy,
        'h1': [],
        'l2': [],
        'mesh_size': []
    },
    {
        'name': 'uniform',
        'strategy': uniform_strategy,
        'h1': [],
        'l2': [],
        'mesh_size': []
    },
    {
        'name': 'adaptive',
        'strategy': adaptive_strategy,
        'h1': [],
        'l2': [],
        'mesh_size': []
    }
]

MAX_MESh_SIZE = 10000

# Dirichlet BC
def boundary(x, on_boundary):
    return on_boundary and (fn.near(x[0], 0) or fn.near(x[1], 0))

for strategy in strategies:
    # Generate initial mesh
    n = 1
    mesh = mshr.generate_mesh(domain, n)
    mesh_size = 0

    while mesh_size < MAX_MESh_SIZE:
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

        bc = fn.DirichletBC(V, 0, boundary)

        # Define forms
        u = fn.TrialFunction(V)
        v = fn.TestFunction(V)

        a = dot(grad(u), grad(v)) * dx
        L = g * v * ds

        # Compute solution
        u_h = fn.Function(V, name='u')
        fn.solve(a == L, u_h, bc)

        # Compute errors
        errorL2 = fn.errornorm(u_exact, u_h, 'L2')
        errorH1 = fn.errornorm(u_exact, u_h, 'H1')
        mesh_size = V.dim()

        strategy['h1'].append(errorH1)
        strategy['l2'].append(errorL2)
        strategy['mesh_size'].append(mesh_size)

        # use markers to create new mesh
        cells_to_refine = strategy['strategy'](mesh, u_exact, u_h)
        mesh = fn.refine(mesh, cells_to_refine)


# plot results.
for strategy in strategies:
    strategy_name = strategy['name']
    mesh_size = strategy['mesh_size']
    h1 = strategy['h1']
    l2 = strategy['l2']

    plt.loglog(mesh_size, h1, label=f'h1 - {strategy_name}')
    plt.loglog(mesh_size, l2, label=f'l2 - {strategy_name}')

plt.legend()
plt.savefig("adapt.png")
