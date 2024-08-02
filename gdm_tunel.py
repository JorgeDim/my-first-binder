'''
Authors: 
    CMM-Mining                                                             
Purpose:                                              
    This script simulates fracture mechanics via a coupled linear
    elasticity - gradient damage model. The total energy is computed
    and thus a variational formulation is then derived.                                                                             
'''
# ========================================
# Import libraries to get the code working
# ========================================
import matplotlib.pyplot as plt
import numpy as np
import sympy 
import socket
import datetime

import dolfinx
import dolfinx.plot
import dolfinx.io 
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

import os, shutil

from Utils.snes_problem import SNESProblem

if MPI.COMM_WORLD.rank == 0:
    print("============================================")
    print(f"This code is built in DOLFINx version: {dolfinx.__version__}")
    print("============================================")

# =============================
# Read mesh from external files
# =============================
if MPI.COMM_WORLD.rank == 0:
    print("=============================")
    print("Read mesh from external files")
    print("=============================")
#malla = "Meshes/cilinder-75_300_150.xdmf"
#malla = "Meshes/mesh_normal.xdmf"
malla = "Meshes/tunel_grueso.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, malla, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# geometry dimension
ndim    = mesh.geometry.dim
fdim    = mesh.topology.dim-1
n       = ufl.FacetNormal(mesh)

# =====================================================
# Material constant. The values of parameters involved
# in the models are defined here
# =====================================================
#E, nu   = dolfinx.fem.Constant(mesh, 2.5e10), dolfinx.fem.Constant(mesh, 0.3)
E, nu   = dolfinx.fem.Constant(mesh, 2.5e10), dolfinx.fem.Constant(mesh, 0.25)
kappa   = dolfinx.fem.Constant(mesh,1.0)
#Gc      = dolfinx.fem.Constant(mesh, 1.0e5)
ell     = dolfinx.fem.Constant(mesh, 0.01) #dolfinx.fem.Constant(mesh, 0.8944)
f       = dolfinx.fem.Constant(mesh,np.array([0,0,0], dtype=np.float64))
a_alpha = "quad"
w_alpha = "quad"


rho         =   2.7e3
g           =   9.8
gravity     =   rho*g

Largo           = np.max(mesh.geometry.x[:,2])-np.min(mesh.geometry.x[:,2])

normal_v    =   ufl.FacetNormal ( mesh)

cargacontrolada = True
condicioninicial = False
# Parameter for minimization
alt_min_parameters = {"atol": 1.e-5,"max_iter": 100}

# ========================
# Create Results Directory
# ======================== 
ahora=datetime.datetime.now().strftime("%y-%m")+'_'+socket.gethostname()
modelname   = "[CargaCont=%s]_[ConInic=%s]_[a_alpha=%s]_[w_alpha=%s]_[kappa=%1.2f]_[np=%d]"%(cargacontrolada,condicioninicial,a_alpha,w_alpha,kappa.value,MPI.COMM_WORLD.Get_size())
if MPI.COMM_WORLD.rank == 0:
    print(modelname)
#savedir     = "results/testigo_%s/%s"%(ahora,modelname)
savedir     = "results/tunel_%s/%s"%(ahora,modelname)
if os.path.isdir(savedir):
    shutil.rmtree(savedir)
# ================================================
# Create function space for 3D elasticity + Damage
# ================================================
# Define the function space for displacement
element_u   = ufl.VectorElement('Lagrange',mesh.ufl_cell(),degree=1,dim=3)
V_u         = dolfinx.fem.FunctionSpace(mesh, element_u)
# Define the function space for damage
element_alpha   = ufl.FiniteElement('Lagrange',mesh.ufl_cell(),degree=1)
V_alpha         = dolfinx.fem.FunctionSpace(mesh, element_alpha)
# Define the state
u           = dolfinx.fem.Function(V_u, name="Displacement")
alpha       = dolfinx.fem.Function(V_alpha, name="Damage")
alpha_dot   = dolfinx.fem.Function(V_alpha, name="Derivative Damage")
state       = {"u": u, "alpha": alpha}
# need upper/lower bound for the damage field
alpha_lb    =   dolfinx.fem.Function(V_alpha, name="Lower bound")
alpha_ub    = dolfinx.fem.Function(V_alpha, name="Upper bound")
# Measure
dx  = ufl.Measure("dx",domain=mesh)

# ======================================================
# Boudary conditions:
# Brief description about the set of boundary conditions 
# for the displacement field.
# ======================================================
def bottom(x):
    return np.isclose(x[2], np.min(x[2]))
def top(x):
    return np.isclose(x[2], np.max(x[2]))
def lateral_x_1(x):
    return np.isclose(x[0], np.min(x[0])) 
def lateral_x_2(x):
    return np.isclose(x[0], np.max(x[0]))
def lateral_y_1(x):
    return np.isclose(x[1], np.min(x[1])) 
def lateral_y_2(x):
    return np.isclose(x[1], np.max(x[1]))
# Boundary facets and dofs for displacement
boundary_facets_top_u   = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, top)
boundary_facets_latx1_u   = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lateral_x_1)
boundary_facets_latx2_u   = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lateral_x_2)
boundary_facets_laty1_u   = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lateral_y_1)
boundary_facets_laty2_u   = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lateral_y_2)

blocked_dofs_top_u      = dolfinx.fem.locate_dofs_topological(V_u.sub(2), mesh.topology.dim-1, boundary_facets_top_u)
blocked_dofs_latx1_u      = dolfinx.fem.locate_dofs_topological(V_u.sub(0), mesh.topology.dim-1, boundary_facets_latx1_u)
blocked_dofs_latx2_u      = dolfinx.fem.locate_dofs_topological(V_u.sub(0), mesh.topology.dim-1, boundary_facets_latx2_u)
blocked_dofs_laty1_u      = dolfinx.fem.locate_dofs_topological(V_u.sub(1), mesh.topology.dim-1, boundary_facets_laty1_u)
blocked_dofs_laty2_u      = dolfinx.fem.locate_dofs_topological(V_u.sub(1), mesh.topology.dim-1, boundary_facets_laty2_u)
blocked_dofs_bottom_u   = dolfinx.fem.locate_dofs_geometrical(V_u, bottom)
# Boundary dofs for damage
blocked_dofs_top_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, top)
blocked_dofs_bottom_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, bottom)
#blocked_dofs_lateral_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, lateral)
# Define values of boundary condition for displacement or load
# Vector value for zero displacement
zero_u = dolfinx.fem.Function(V_u)
with zero_u.vector.localForm() as bc_local:
    bc_local.set(0.0)
#zero_u.vector.destroy()
# Scalar value for non-zero load
nonzero_load = dolfinx.fem.Constant(mesh, ScalarType(1.0))
# Scalar value for non-zero damage
one_alpha = dolfinx.fem.Function(V_alpha)
with one_alpha.vector.localForm() as bc_local:
    bc_local.set(1.0)
# Scalar value for zero damgage                 
zero_alpha = dolfinx.fem.Function(V_alpha)
with zero_alpha.vector.localForm() as bc_local:
    bc_local.set(0.0)
# Define the Dirichlet boundary conditions for displacement
nonzero_u = dolfinx.fem.Constant(mesh, ScalarType(1.0))
bc_u0 = dolfinx.fem.dirichletbc(zero_u, blocked_dofs_bottom_u)
bc_u1 = dolfinx.fem.dirichletbc(nonzero_u, blocked_dofs_top_u, V_u.sub(2))
bc_u2 = dolfinx.fem.dirichletbc(0.0,blocked_dofs_latx1_u, V_u.sub(0))
bc_u3 = dolfinx.fem.dirichletbc(0.0,blocked_dofs_latx2_u, V_u.sub(0))
bc_u4 = dolfinx.fem.dirichletbc(0.0,blocked_dofs_laty1_u, V_u.sub(1))
bc_u5 = dolfinx.fem.dirichletbc(0.0,blocked_dofs_laty2_u, V_u.sub(1))
# Define the Dirichlet boundary conditions for damage
bc_alpha0 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_top_alpha)
bc_alpha1 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_bottom_alpha)
#bc_alpha2 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_lateral_alpha)
# Merge the boundary condition for displacement and damage
if cargacontrolada:
    bcs_u = [bc_u0]
else:
    bcs_u = [bc_u0,bc_u1]

#bcs_alpha = [bc_alpha0,bc_alpha1]
bcs_alpha = [ ]
# setting the upper bound to 0 where BCs are applied
alpha_ub.interpolate(one_alpha)
dolfinx.fem.set_bc(alpha_ub.vector, bcs_alpha)
# Define Neumann bondary conditions
# Set markers and locations of boundaries
if cargacontrolada:
    boundaries = [(1,top),(2,bottom),(3,lateral_x_1),(4,lateral_x_2),(5,lateral_y_1),(6,lateral_y_2)]
    facet_indices, facet_markers = [],[]
    for (marker, locator) in boundaries:
        facets =  dolfinx.mesh.locate_entities_boundary(mesh,fdim,locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets,marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag =  dolfinx.mesh.meshtags(mesh,fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    # Define new measure including boundary naming
    # top: ds(1), bottom:  ds(2)
    # lateral : ds(3)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
else:
    ds=ufl.Measure("ds", domain=mesh)
def initial_condition(x):
     a_1=0.002
     a_2=0.002
     a_3=0.00001
     term_1 = ((np.cos(np.pi/4)*(x[0])+np.sin(np.pi/4)*(x[2]))**2)/a_1
     term_2 = (x[1]**2)/a_2
     term_3 = ((-np.sin(np.pi/4)*(x[0])+np.cos(np.pi/4)*(x[2]))**2)/a_3
     if condicioninicial:
        return 1.0*((term_1+term_2+term_3)<=1)
     else:
         return 0.0*((term_1+term_2+term_3)<=1)
     
#def g_bc_zz(x):
#    lmbda       =   E * nu / ( 1.0 - nu**2)
#    ffD         =   lmbda/(lmbda+2*mu)
#    k = gravity
#    bound_condition = ffD * k * (x[2]-Largo)
#    return bound_condition
#g_bc= g_bc_zz  


lmbda       =   E * nu / ( 1.0 - nu**2)
mu          =   E / ( 2.0 * ( 1.0 + nu))
k = gravity

x = ufl.SpatialCoordinate(mesh)
g_bc_zz =  ufl.exp (  k * (x[2]-Largo))
# =====================================================
# In this block of code define the operators.  These is
# independent from the mesh.
# -----------------------------------------------------
# Constitutive functions of the damage model. Here
# we define the operators acting on the damage system
# as well as the operator acting on the displacement
# field, which depends on the damage.
# =====================================================
def w(alpha):
    """Dissipated energy function as a function of the damage """
    if w_alpha == "lin":
        return alpha
    if w_alpha == "quad":
        return alpha**2
def a(alpha, k_ell=1.e-6):
    """Stiffness modulation as a function of the damage """
    if a_alpha == "lin":
        return (1 - alpha) + k_ell
    if a_alpha == "quad":
        return (1 - alpha) ** 2 + k_ell
def a_der(alpha):
    """Derivative of a(alpha)"""
    if a_alpha == "lin":
        return -1
    if a_alpha == "quad":
        return -2*(1 - alpha)
    if a_alpha == "fraq":
        return -1.0/((1+alpha)**2)
def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))
def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    mu    = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / (1.0 - nu ** 2)
    return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)
def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(alpha) * sigma_0(u)
def sigma_der(u,alpha):
    """derivada del tensor de tensiones del material dañado en función del desplazamiento y del daño"""
    return a_der(alpha) * sigma_0(u)
#  Define th Deviator and Spheric Tensors
def Dev(Tensor):
    return Tensor-Sph(Tensor)
def Sph(Tensor):  
    Tensor2=ufl.Identity(ndim) 
    return (ufl.inner(Tensor,Tensor2)/ufl.inner(Tensor2,Tensor2))*Tensor2   

# =====================================================
# Constants
# =====================================================
z       = sympy.Symbol("z")
c_w     = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
Gc      = dolfinx.fem.Constant(mesh, float(c_w)*3577)
#print("c_w = ",c_w)
c_1w    = sympy.integrate(sympy.sqrt(1/w(z)),(z,0,1))
#print("c_1/w = ",c_1w)
tmp     = 2*(sympy.diff(w(z),z)/sympy.diff(1/a(z),z)).subs({"z":0})
sigma_c = sympy.sqrt(tmp * Gc.value * E.value / (c_w * ell.value))
#print("sigma_c = %2.3f"%sigma_c)
eps_c   = float(sigma_c/E.value)
#print("eps_c = %2.3f"%eps_c)

# =====================================================
# Useful functions for minimization
# =====================================================
def simple_monitor(state, iteration, error_L2):
    alpha       = state["alpha"]
    alpha_max   = np.amax(alpha.x.array)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Iteration: {iteration:3d}, Error: {error_L2:3.4e}, Max Alpha: {alpha_max:.3f}")
def alternate_minimization(state,problem_u,solver_alpha_snes,parameters=alt_min_parameters,monitor=None):
    u           = state["u"]
    alpha       = state["alpha"]
    alpha_old   = dolfinx.fem.Function(alpha.function_space)
    # Set previous alpha as alpha_old
    alpha.vector.copy(alpha_old.vector)
    for iteration in range(parameters["max_iter"]):                 
        # solve displacement
        problem_u.solve()
        # solve damage
        solver_alpha_snes.solve(None, alpha.vector)
        # check error and update
        L2_error    = dolfinx.fem.form(ufl.inner(alpha - alpha_old, alpha - alpha_old) * dx)
        error_L2    = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error),op=MPI.SUM))
        # uptade alpha_old with the value of alpha
        alpha.vector.copy(alpha_old.vector)
        # Monitor of solutions
        if monitor is not None:
            monitor(state, iteration, error_L2)
        # check error                      
        if error_L2 <= parameters["atol"]:
            break
    else:
        pass #raise RuntimeError(f"Could not converge after {iteration:3d} iteration, error {error_L2:3.4e}") 
    return (error_L2, iteration)
def postprocessing(state, iteration, error_L2):
    # Save number of iterations for the time step
    u = state["u"]
    alpha = state["alpha"]
    iterations[i_t] = np.array([t,i_t])
    # Compute integrals
    vol             = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * dx))
    #Largo           = np.max(x[2])-np.min(x[2])
    int_alpha_temp  = alpha * dx
    int_alpha       = dolfinx.fem.assemble_scalar(dolfinx.fem.form(int_alpha_temp))
    alpha_max       = np.amax(alpha.x.array)
    int_sigmazz_temp  = sigma(u,alpha)[2,2]/vol * dx
    int_sigmazz       = dolfinx.fem.assemble_scalar(dolfinx.fem.form(int_sigmazz_temp))
    if cargacontrolada:
        int_epszz_temp  = eps(u)[2,2]/vol * dx
        epsilon_maquina      = dolfinx.fem.assemble_scalar(dolfinx.fem.form(int_epszz_temp))
    else:
        epsilon_maquina = nonzero_u/Largo
    # Compute energies
    elastic_energy_value    = dolfinx.fem.assemble_scalar(dolfinx.fem.form(elastic_energy))
    surface_energy_value    = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dissipated_energy))
    energies[i_t]           = np.array([i_t,elastic_energy_value,surface_energy_value,elastic_energy_value+surface_energy_value])
    ints_domains[i_t]       = np.array([i_t,int_alpha/vol,abs(int_sigmazz),abs(epsilon_maquina),alpha_max])
    simple_monitor(state, iteration, error_L2)

# =====================================================
# Let us define the total energy of the system as the 
# sum of elastic energy, dissipated energy due to the 
# damage and external work due to body forces. 
# =====================================================
elastic_energy      = 0.5 * ufl.inner(sigma(u,alpha), eps(u)) * dx 
elastic_energy2 =   0.5*(ufl.inner(Dev(sigma(u,alpha)),Dev(eps(u))) \
                        +(1-kappa)*ufl.inner(Sph(sigma(u,alpha)),Sph(eps(u))))*dx
dissipated_energy   = Gc / float(c_w) * (w(alpha) / ell + ell * ufl.dot(ufl.grad(alpha), ufl.grad(alpha))) * dx
external_work       = ufl.dot(f, u) * dx
if cargacontrolada: 
    bounbdary_energy      = ufl.dot(nonzero_load*n,u)*ds(1)+0.5*(ufl.dot(g_bc_zz*n,u)*ds(3) 
                                                        +ufl.dot(g_bc_zz*normal_v,u)*ds(4) 
                                                        +ufl.dot(g_bc_zz*normal_v,u)*ds(5) 
                                                         +ufl.dot(g_bc_zz*normal_v,u)*ds(6))
    total_energy        = elastic_energy + dissipated_energy - external_work - bounbdary_energy
    #total_energy2        = elastic_energy2 + dissipated_energy - external_work - bounbdary_energy
else:
    total_energy        = elastic_energy + dissipated_energy - external_work 



total_disip         =  0.5*kappa*ufl.inner(Sph( sigma_der(u,alpha)),Sph( eps(u)))*alpha_dot*dx

# =====================================================
# Weak form of elasticity problem. This is the formal 
# expression for the tangent problem which gives us the 
# equilibrium equations
# =====================================================
E_u             = ufl.derivative(total_energy,u,ufl.TestFunction(V_u))
E_du            = ufl.replace(E_u,{u: ufl.TrialFunction(V_u)})
E_alpha         = ufl.derivative(total_energy,alpha,ufl.TestFunction(V_alpha))
#E_alpha         = ufl.derivative(total_energy2,alpha,ufl.TestFunction(V_alpha))
Dis_alphadot    = ufl.derivative(total_disip,alpha_dot,ufl.TestFunction(V_alpha))
#EE_alpha        = ufl.derivative(elastic_energy,alpha,ufl.TestFunction(V_alpha))
#E_alpha_alpha   = ufl.derivative(E_alpha,alpha,ufl.TrialFunction(V_alpha))
E_alpha_alpha   = ufl.derivative(E_alpha+Dis_alphadot,alpha,ufl.TrialFunction(V_alpha))
jacobian        = dolfinx.fem.form(E_alpha_alpha)
#residual        = dolfinx.fem.form(E_alpha)
residual        = dolfinx.fem.form(E_alpha+Dis_alphadot)
# Displacement problem
problem_u   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_du), L=ufl.rhs(E_du), bcs=bcs_u, u=u,
                                      petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
# Damage problem
damage_problem      = SNESProblem(residual, jacobian, alpha, bcs_alpha)
b                   = dolfinx.cpp.la.petsc.create_vector(V_alpha.dofmap.index_map, V_alpha.dofmap.index_map_bs)
J                   = dolfinx.fem.petsc.create_matrix(damage_problem.a)
solver_alpha_snes   = PETSc.SNES().create()
solver_alpha_snes.setType("vinewtonrsls")
solver_alpha_snes.setFunction(damage_problem.F, b)
solver_alpha_snes.setJacobian(damage_problem.J, J)
solver_alpha_snes.setTolerances(rtol=1.0e-8, max_it=50)
solver_alpha_snes.getKSP().setType("cg")
solver_alpha_snes.getKSP().setTolerances(rtol=1.0e-8)
solver_alpha_snes.getKSP().getPC().setType("jacobi")
# We set the bound (Note: they are passed as reference and not as values)
solver_alpha_snes.setVariableBounds(alpha_lb.vector,alpha_ub.vector)
#solver_alpha_snes.setVariableBounds(alpha_lb.vector,1.0e20*alpha_ub.vector)
# reference value for the loading 
load0   = 1.0
#loads   = load0*np.linspace(0.,1.,40)
loads   = load0*np.linspace(0.,1.,20)
# Create array to save some results
energies        = np.zeros((len(loads),4))
iterations      = np.zeros((len(loads),2))
ints_domains    = np.zeros((len(loads),5))
# Set initial condition for damage
#with alpha.vector.localForm() as alpha_local:
#    alpha_local.set(0)
alpha.interpolate(initial_condition)
# Crete the files to store the solutions
u_file      = dolfinx.io.VTKFile(mesh.comm, savedir+"/u.pvd", "w")
alpha_file  = dolfinx.io.VTKFile(mesh.comm, savedir+"/alpha.pvd", "w")
u_file.write_mesh(mesh) 
alpha_file.write_mesh(mesh)  
load = 0
for i_t, t in enumerate(loads):
    
    if cargacontrolada:
        load=-4.3e7*t
        nonzero_load.value=load 
    else:
        load=-0.0004*t
        nonzero_u.value=load 
    # update the lower bound
    alpha.vector.copy(alpha_lb.vector)  
    if MPI.COMM_WORLD.rank == 0:  
        print(f"-- Solving for t = {t:3.4f} --")
        print(f"-- Load = {load:2.7f} --")
    # alternate minimization
    alternate_minimization(state,problem_u,solver_alpha_snes,parameters=alt_min_parameters,monitor=postprocessing)
    # save solutions
    u_file.write_function(u,t)
    alpha_file.write_function(alpha,t)
u_file.close()
alpha_file.close()

# =====================================================
# Plots
# =====================================================
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

t = ints_domains[:,3]
data1 = ints_domains[:,2]
data2 = ints_domains[:,4]
fig, ax1 = plt.subplots()
color = 'blue'

ax1.set_xlabel(r'$\vert \varepsilon_{zz} \vert$')
ax1.set_ylabel(r'$\vert \sigma_{zz}(\varepsilon)\vert $', color=color)
ax1.plot(t, data1,'-',color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()

color = 'red'
ax2.set_ylabel(r'$ \alpha_{max}(\varepsilon) $', color=color)
ax2.plot(t, data2,'-', color=color)
ax2.tick_params(axis='y', labelcolor=color)
#plt.gca().invert_xaxis()
plt.grid()
plt.savefig(savedir+"/sigma_eps.png")
plt.show()
plt.clf()


# Plot integrate of damage
p1, = plt.plot(ints_domains[:,3], ints_domains[:,2],'b*-',linewidth=2)
p2, = plt.plot(ints_domains[:,3], 4.3e7*ints_domains[:,4],'r*-',linewidth=2)
plt.legend([p1,p2], ["sigma(epsilon)","alpha_max(epsilon)"])
plt.gca().invert_xaxis()
plt.grid()
plt.xlabel('epsilon')
plt.ylabel('sigma(b),alpha(r)')
#plt.savefig(savedir+"/sigma_eps.png")
plt.clf()

# Plot integrate of damage
p1, = plt.plot(ints_domains[:,0], ints_domains[:,1],'r*-',linewidth=2)
plt.legend([p1], ["Damage_int"])
plt.xlabel('iteration')
plt.ylabel('Damage')
plt.savefig(savedir+"/Damage_int.png")
plt.show()
plt.clf()

# Plot Energies vs Displacement
p1, = plt.plot(energies[:,0], energies[:,1],'b*',linewidth=2)
p2, = plt.plot(energies[:,0], energies[:,2],'r^',linewidth=2)
p3, = plt.plot(energies[:,0], energies[:,3],'ko',linewidth=2)
plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"])
plt.xlabel('iteration')
plt.ylabel('Energies')
plt.savefig(savedir+"/energies.png")

fDatos      =   open(savedir + '/Parametros.txt', 'w')
buff='A_alpha   =%s\n'%(a_alpha);fDatos.write(buff)     ; print(buff)
buff='w_alpha   =%s\n'%(w_alpha);fDatos.write(buff)     ; print(buff)
buff='kappa     =%f\n'%(kappa);fDatos.write(buff); print (buff)
buff='E         =%1.1e\n'%(E.value);fDatos.write(buff) ; print (buff)
buff='nu        =%f\n'%(nu.value);fDatos.write(buff)     ; print (buff)
buff='ell       =%f\n'%(ell.value);fDatos.write(buff)   ; print (buff)
buff='c_w       =%f\n'%(float(c_w));fDatos.write(buff)     ; print (buff)
buff='G_c       =%f\n'%(Gc);fDatos.write(buff); print (buff)
fDatos.close()
