'''
Authors: 
    CMM-Mining                                                             
Purpose:                                              
    This script simulates fracture mechanics via a coupled linear
    elasticity - gradient damage model. The total energy is computed
    and thus a variational formulation is then derived.                                                                             
'''

# ================================================================================
# Parameters for differents examples
# ================================================================================

presion_lateral=0
sin_tunel=0
material = 1

density = 2.7e3
grav_acel = 9.8

condicioninicial = False

if material ==1:
    young=76.e9
    poisson=0.37
    C_ell = 10.


    C_1=-0.31
    C_2=1.9448411e5/60/10
    C_3=-198.94

if material ==2:
    '''Presentación 24 de abril pag.12'''
    young=24.07e9
    poisson=0.37
    C_ell = 10.


    C_1=-0.3744
    C_2=95770
    C_3=420254


# Create mesh and define function space
L = 100.0
altura_0=1000.0



# Parameter for minimization
alt_min_parameters = {"atol": 1.e-5,"max_iter": 100}

# ================================================================================
# Import libraries to get the code working
# ================================================================================

import dolfinx.cpp
import dolfinx.fem.forms
import dolfinx.fem.function
import dolfinx.fem.petsc
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

import ufl.algebra
import ufl.classes

from Utils.snes_problem import SNESProblem

if MPI.COMM_WORLD.rank == 0:
    print("============================================")
    print(f"This code is built in DOLFINx version: {dolfinx.__version__}")
    print("============================================")


# ================================================================================
# Read mesh from external files
# ================================================================================
if MPI.COMM_WORLD.rank == 0:
    print("=============================")
    print("Read mesh from external files")
    print("=============================")

if sin_tunel:
    mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, L, L]], [50, 50, 50], dolfinx.mesh.CellType.tetrahedron)
else:
    malla = "Meshes/tunel.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, malla, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    

# geometry dimension
ndim    = mesh.geometry.dim
fdim    = mesh.topology.dim-1
n       = ufl.FacetNormal(mesh) 
x = ufl.SpatialCoordinate(mesh)
# ================================================================================
# Material constant. The values of parameters involved
# in the models are defined here
# ================================================================================
E, nu   = dolfinx.fem.Constant(mesh, young), dolfinx.fem.Constant(mesh, poisson)
kappa   = dolfinx.fem.Constant(mesh,1.0)
ell     = dolfinx.fem.Constant(mesh, C_ell) 


altura = dolfinx.fem.Constant(mesh,altura_0)

rho = dolfinx.fem.Constant(mesh,density)
gg  = dolfinx.fem.Constant(mesh,grav_acel)

Profundidad           = np.max(mesh.geometry.x[:,2]) + altura 

f = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, -rho*gg)))

g_bc_zz =  lambda x: rho*gg*(x[2]-Profundidad) 
g_bc = g_bc_zz(x)
# ================================================================================
# Create Results Directory
# ================================================================================
ahora=datetime.datetime.now().strftime("%y-%m-%d")
modelname   = "[ConInic=%s]_[kappa=%1.2f]_[np=%d]"%(condicioninicial,kappa.value,MPI.COMM_WORLD.Get_size())

savedir     = "Results_Tunel_%s/%s"%(ahora,modelname)
if MPI.COMM_WORLD.rank == 0:
    print(f"Save in: {savedir:s}")

if os.path.isdir(savedir):
    shutil.rmtree(savedir)

# ================================================================================
# Create function space for 3D elasticity + Damage
# ================================================================================
# Define the function space for displacement
V_vec = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
# Define the function space for damage
V_sca = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# Define the state
u           = dolfinx.fem.Function(V_vec, name="Displacement")
u_0           = dolfinx.fem.Function(V_vec)
alpha       = dolfinx.fem.Function(V_sca, name="Damage")
alpha_old   = dolfinx.fem.Function(alpha.function_space, name="Damage_old")
state       = {"u": u, "alpha": alpha,"alpha_old":alpha_old}
# need upper/lower bound for the damage field
alpha_lb    =   dolfinx.fem.Function(V_sca, name="Lower bound")
alpha_ub    = dolfinx.fem.Function(V_sca, name="Upper bound")

# ================================================================================
# Boudary conditions:
# Brief description about the set of boundary conditions 
# for the displacement field.
# ================================================================================
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

# Boundary dofs for damage
blocked_dofs_top_alpha = dolfinx.fem.locate_dofs_geometrical(V_sca , top)
blocked_dofs_bottom_alpha = dolfinx.fem.locate_dofs_geometrical(V_sca , bottom)
blocked_dofs_lateral_x_1_alpha = dolfinx.fem.locate_dofs_geometrical(V_sca , lateral_x_1)
blocked_dofs_lateral_x_2_alpha = dolfinx.fem.locate_dofs_geometrical(V_sca , lateral_x_2)
blocked_dofs_lateral_y_1_alpha = dolfinx.fem.locate_dofs_geometrical(V_sca , lateral_y_1)
blocked_dofs_lateral_y_2_alpha = dolfinx.fem.locate_dofs_geometrical(V_sca , lateral_y_2)
# Scalar value for zero damgage   
zero_alpha = np.array(0.0, dtype=dolfinx.default_scalar_type)
one_alpha  = np.array(1.0, dtype=dolfinx.default_scalar_type)
# Define the Dirichlet boundary conditions for damage
bc_alpha0 = dolfinx.fem.dirichletbc(zero_alpha,blocked_dofs_top_alpha,V_sca)
bc_alpha1 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_bottom_alpha,V_sca)
bc_alpha2 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_lateral_x_1_alpha,V_sca) 
bc_alpha3 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_lateral_x_2_alpha,V_sca)
bc_alpha4 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_lateral_y_1_alpha,V_sca)
bc_alpha5 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_lateral_y_2_alpha,V_sca)
# Merge the boundary condition for damage
if sin_tunel:
    bcs_alpha = [bc_alpha0,bc_alpha1,bc_alpha2,bc_alpha3,bc_alpha4,bc_alpha5]
else:
    bcs_alpha = [bc_alpha0,bc_alpha1,bc_alpha3,bc_alpha4,bc_alpha5]


# Boundary facets and dofs for displacement
boundary_facets_bottom_u   = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom)
blocked_dofs_bottom_u   = dolfinx.fem.locate_dofs_topological(V_vec.sub(2), fdim, boundary_facets_bottom_u)
boundary_facets_lateral_x_1_u   = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lateral_x_1)
blocked_dofs_lateral_x_1_u   = dolfinx.fem.locate_dofs_topological(V_vec.sub(0), fdim, boundary_facets_lateral_x_1_u)
boundary_facets_lateral_x_2_u   = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lateral_x_2)
blocked_dofs_lateral_x_2_u   = dolfinx.fem.locate_dofs_topological(V_vec.sub(0), fdim, boundary_facets_lateral_x_2_u)
boundary_facets_lateral_y_1_u   = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lateral_y_1)
blocked_dofs_lateral_y_1_u   = dolfinx.fem.locate_dofs_topological(V_vec.sub(1), fdim, boundary_facets_lateral_y_1_u)
boundary_facets_lateral_y_2_u   = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lateral_y_2)
blocked_dofs_lateral_y_2_u   = dolfinx.fem.locate_dofs_topological(V_vec.sub(1), fdim, boundary_facets_lateral_y_2_u)
# Scalar value for zero displacement  
zero_un = np.array(0.0, dtype=dolfinx.default_scalar_type)

# Define the Dirichlet boundary conditions for displacement
bc_u0 = dolfinx.fem.dirichletbc(zero_un, blocked_dofs_bottom_u,V_vec.sub(2))
bc_u1 = dolfinx.fem.dirichletbc(zero_un, blocked_dofs_lateral_x_1_u,V_vec.sub(0))
bc_u2 = dolfinx.fem.dirichletbc(zero_un, blocked_dofs_lateral_x_2_u,V_vec.sub(0))
bc_u3 = dolfinx.fem.dirichletbc(zero_un, blocked_dofs_lateral_y_1_u,V_vec.sub(1))
bc_u4 = dolfinx.fem.dirichletbc(zero_un, blocked_dofs_lateral_y_2_u,V_vec.sub(1))
# Merge the boundary condition for displacement
if presion_lateral:
    bcs_u  = [bc_u0]
else:
    bcs_u  = [bc_u0,bc_u1,bc_u2,bc_u3,bc_u4]

# Define Neumann bondary conditions
# Set markers and locations of boundaries
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
# Measure
dx  = ufl.Measure("dx",domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)



# =====================================================
# INITIAL CONDITION
# =====================================================

# setting the upper bound to 0 where BCs are applied

#alpha_ub=np.array(1.0, dtype=dolfinx.default_scalar_type)
#alpha_lb=np.array(0.0, dtype=dolfinx.default_scalar_type)
alpha_ub.x.array[:] = 1.0
alpha_lb.x.array[:] = 0.0
dolfinx.fem.set_bc(alpha_ub.vector, bcs_alpha)

def initial_condition(x):
     a_1=0.002
     a_2=0.002
     a_3=0.00001
     term_1 = ((np.cos(np.pi/4)*(x[0])+np.sin(np.pi/4)*(x[2]))**2)/a_1
     term_2 = (x[1]**2)/a_2
     term_3 = ((-np.sin(np.pi/4)*(x[0])+np.cos(np.pi/4)*(x[2]))**2)/a_3
     if condicioninicial:
        return 1.0*((term_1+term_2+term_3)<=30)
     else:
         return 0.0*((term_1+term_2+term_3)<=30)

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
    return C_2*alpha + C_3*alpha**2
def a(alpha, k_ell=1.e-6):
    """Stiffness modulation as a function of the damage """
    return 1-(1+C_1)*alpha+C_1*alpha**2
def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))
def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    mu    = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu)*(1-2*nu)) 
    return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)
def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(alpha) * sigma_0(u)
#  Define th Deviator and Spheric Tensors
def Dev(Tensor):
    return Tensor-Sph(Tensor)
def Sph(Tensor):  
    #Tensor2=ufl.Identity(ndim) 
    Tensor2=ufl.as_matrix([[nu/(1-nu),0,0],[0,nu/(1-nu),0],[0,0,1]])
    return (ufl.inner(Tensor,Tensor2)/ufl.inner(Tensor2,Tensor2))*Tensor2

# ================================================================================
# Save Files 
# ================================================================================
u_file                  = dolfinx.io.XDMFFile(mesh.comm, savedir+"/u.xdmf", "w")
alpha_file              = dolfinx.io.XDMFFile(mesh.comm, savedir+"/alpha.xdmf", "w")
sigma_file              = dolfinx.io.XDMFFile(mesh.comm, savedir+"/sigma.xdmf", "w")
eps_file                = dolfinx.io.XDMFFile(mesh.comm, savedir+"/epsilon.xdmf", "w")
sigma_vonmisses_file    = dolfinx.io.XDMFFile(mesh.comm, savedir+"/sigma_vonmisses.xdmf", "w")
sigma_desviador_file    = dolfinx.io.XDMFFile(mesh.comm, savedir+"/sigma_desviador.xdmf", "w")
energia_alpha_file      = dolfinx.io.XDMFFile(mesh.comm, savedir+"/energia_alpha.xdmf", "w")
u_file.write_mesh(mesh)
alpha_file.write_mesh(mesh)  
sigma_file.write_mesh(mesh)
eps_file.write_mesh(mesh)
energia_alpha_file.write_mesh(mesh) 
sigma_vonmisses_file.write_mesh(mesh)
sigma_desviador_file.write_mesh(mesh)
# ================================================================================
# Useful functions 
# ================================================================================

def proyectar_escalar(escalar):
    escalar_temp=dolfinx.fem.Function(V_sca ,name="escalar")
    a_tmp = ufl.inner(ufl.TrialFunction(V_sca ), ufl.TestFunction(V_sca )) * ufl.dx
    L_tmp = ufl.inner(escalar, ufl.TestFunction(V_sca )) * ufl.dx
    problem_tmp = dolfinx.fem.petsc.LinearProblem(a_tmp, L_tmp,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    escalar_temp = problem_tmp.solve()
    return escalar_temp

def proyectar_vector(vec):
    tmp=dolfinx.fem.Function(V_vec ,name="tmp")
    a_tmp = ufl.inner(ufl.TrialFunction(V_vec ), ufl.TestFunction(V_vec )) * ufl.dx
    L_tmp = ufl.inner(vec, ufl.TestFunction(V_vec )) * ufl.dx
    problem_tmp = dolfinx.fem.petsc.LinearProblem(a_tmp, L_tmp,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    tmp = problem_tmp.solve()
    return tmp

def norma_L2(funcion):
    L2_error =dolfinx.fem.form(ufl.inner(funcion, funcion) * dx)
    return np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error),op=MPI.SUM))

def guardar(u,alpha,t,densidad_energy2):
    print(f"Guardando Soluciones t={t:d}")
    print(f"Max_alpha: {np.max(alpha.x.array):.5f}, Min_alpha: {np.min(alpha.x.array):.5f}\n")
    print(f"Max_U[2]: {np.max(u[2].x.array):.5f}, Min_U[2]: {np.min(u[2].x.array):.5f}\n")
    u_file.write_function(u,t)
    alpha_file.write_function(alpha,t)

    energia_alpha=dolfinx.fem.Function(V_sca ,name="energia_alpha")

    a_tmp = ufl.inner(ufl.TrialFunction(V_sca), ufl.TestFunction(V_sca)) * ufl.dx
    L_tmp = ufl.inner(densidad_energy2, ufl.TestFunction(V_sca)) * ufl.dx
    problem_tmp = dolfinx.fem.petsc.LinearProblem(a_tmp, L_tmp,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    energia_alpha = problem_tmp.solve()
    
    energia_alpha_file.write_function(energia_alpha,t)
    
    # Define the tensorial functional space
    V_ten = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,mesh.geometry.dim))) #considerar esa
    #V_eps = dolfinx.fem.TensorFunctionSpace(mesh, ("Lagrange", 1))
    #V_sig = dolfinx.fem.TensorFunctionSpace(mesh, ("Lagrange", 1))
    #Define the functions for stresses and deformations
    eps_t     = dolfinx.fem.Function(V_ten,name="epsilon")
    sigma_t     = dolfinx.fem.Function(V_ten,name="sigma")
    # Project deformation and stress
    eps_expr = eps(u)
    sig_expr = sigma(u,alpha)
    # Strain
    a_eps = ufl.inner(ufl.TrialFunction(V_ten), ufl.TestFunction(V_ten)) * ufl.dx
    L_eps = ufl.inner(eps_expr, ufl.TestFunction(V_ten)) * ufl.dx
    problem_eps = dolfinx.fem.petsc.LinearProblem(a_eps, L_eps,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    eps_t = problem_eps.solve()
    # Stress
    a_sig = ufl.inner(ufl.TrialFunction(V_ten), ufl.TestFunction(V_ten)) * ufl.dx
    L_sig = ufl.inner(sig_expr, ufl.TestFunction(V_ten)) * ufl.dx
    problem_sig = dolfinx.fem.petsc.LinearProblem(a_sig, L_sig,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    sigma_t = problem_sig.solve()
    # Save tresses and strain
    sigma_file.write_function(sigma_t,t)
    eps_file.write_function(eps_t,t)


    s = Dev(sigma(u,alpha))
    von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))

    V_von_mises = dolfinx.fem.functionspace(mesh, ("DG", 0))
    stress_expr = dolfinx.fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
    stresses = dolfinx.fem.Function(V_von_mises)
    stresses.interpolate(stress_expr)

    a_sig_desviador = ufl.inner(ufl.TrialFunction(V_ten), ufl.TestFunction(V_ten)) * ufl.dx
    L_sig_desviador = ufl.inner(s, ufl.TestFunction(V_ten)) * ufl.dx
    problem_sig_desviador = dolfinx.fem.petsc.LinearProblem(a_sig_desviador, L_sig_desviador,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    sigma_desviador = problem_sig_desviador.solve()

    sigma_vonmisses_file.write_function(stresses,t)
    sigma_desviador_file.write_function(sigma_desviador,t)

    densidad_energy2_max   = np.amax(energia_alpha.x.array)

    sigma_22 = proyectar_escalar(sigma_t[2,2])
    epsilon_22= proyectar_escalar(eps_t[2,2])
    sigma_22_max   = np.max(sigma_22.x.array)
    sigma_22_min   = np.min(sigma_22.x.array)
    epsilon_22_max   = np.max(epsilon_22.x.array)
    epsilon_22_min   = np.min(epsilon_22.x.array)

    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Iteration: {t:3d}, Max energia_alpha  : {densidad_energy2_max:.3e}, C_2 : {C_2:.3e}, R : {densidad_energy2_max/C_2:.5f}")
        print(f"Iteration: {t:3d}, Norma energia_alpha: {norma_L2(energia_alpha):.3e}")
        print(f"Iteration: {t:3d}, Min sigma          : {sigma_22_min:.3e}")
        print(f"Iteration: {t:3d}, Max sigma          : {sigma_22_max:.3e}")
        print(f"Iteration: {t:3d}, Min eps            : {epsilon_22_min:.3e}")
        print(f"Iteration: {t:3d}, Max eps            : {epsilon_22_max:.3e}")


def guardar_2(u,alpha,t,densidad_energy2):
    u_file.write_function(u,t)
    alpha_file.write_function(alpha,t)

    energia_alpha=dolfinx.fem.Function(V_sca ,name="energia_alpha")

    a_tmp = ufl.inner(ufl.TrialFunction(V_sca), ufl.TestFunction(V_sca)) * ufl.dx
    L_tmp = ufl.inner(densidad_energy2, ufl.TestFunction(V_sca)) * ufl.dx
    problem_tmp = dolfinx.fem.petsc.LinearProblem(a_tmp, L_tmp,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    energia_alpha = problem_tmp.solve()
    
    energia_alpha_file.write_function(energia_alpha,t)
    
    # Define the tensorial functional space
    V_ten = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,mesh.geometry.dim))) #considerar esa
    #V_eps = dolfinx.fem.TensorFunctionSpace(mesh, ("Lagrange", 1))
    #V_sig = dolfinx.fem.TensorFunctionSpace(mesh, ("Lagrange", 1))
    #Define the functions for stresses and deformations
    eps_t     = dolfinx.fem.Function(V_ten,name="epsilon")
    sigma_t     = dolfinx.fem.Function(V_ten,name="sigma")
    # Project deformation and stress
    eps_expr = eps(u)
    sig_expr = sigma(u,alpha)
    # Strain
    a_eps = ufl.inner(ufl.TrialFunction(V_ten), ufl.TestFunction(V_ten)) * ufl.dx
    L_eps = ufl.inner(eps_expr, ufl.TestFunction(V_ten)) * ufl.dx
    problem_eps = dolfinx.fem.petsc.LinearProblem(a_eps, L_eps,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    eps_t = problem_eps.solve()
    # Stress
    a_sig = ufl.inner(ufl.TrialFunction(V_ten), ufl.TestFunction(V_ten)) * ufl.dx
    L_sig = ufl.inner(sig_expr, ufl.TestFunction(V_ten)) * ufl.dx
    problem_sig = dolfinx.fem.petsc.LinearProblem(a_sig, L_sig,petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-8"})
    sigma_t = problem_sig.solve()
    # Save tresses and strain
    sigma_file.write_function(sigma_t,t)
    eps_file.write_function(eps_t,t)


    densidad_energy2_max   = np.amax(energia_alpha.x.array)

    sigma_22 = proyectar_escalar(sigma_t[2,2])
    epsilon_22= proyectar_escalar(eps_t[2,2])
    sigma_22_max   = np.max(sigma_22.x.array)
    sigma_22_min   = np.min(sigma_22.x.array)
    epsilon_22_max   = np.max(epsilon_22.x.array)
    epsilon_22_min   = np.min(epsilon_22.x.array)

    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Iteration: {t:3d}, Max energia_alpha  : {densidad_energy2_max:.3e}, C_2 : {C_2:.3e}, R : {densidad_energy2_max/C_2:.5f}")
        print(f"Iteration: {t:3d}, Norma energia_alpha: {norma_L2(energia_alpha):.3e}")
        print(f"Iteration: {t:3d}, Min sigma          : {sigma_22_min:.3e}")
        print(f"Iteration: {t:3d}, Max sigma          : {sigma_22_max:.3e}")
        print(f"Iteration: {t:3d}, Min eps            : {epsilon_22_min:.3e}")
        print(f"Iteration: {t:3d}, Max eps            : {epsilon_22_max:.3e}")

# ================================================================================
# Useful functions for minimization
# ================================================================================
def simple_monitor(state, iteration, error_L2):
    alpha       = state["alpha"]
    alpha_old   = state["alpha_old"]
    alpha_max   = np.max(alpha.x.array)
    alpha_min   = np.min(alpha.x.array)
    alpha_old_max   = np.max(alpha_old.x.array)
    alpha_old_min   = np.min(alpha_old.x.array)
    if MPI.COMM_WORLD.rank == 0:
        #print(f"Iteration: {iteration:3d}, Error: {error_L2:3.4e}, Max Alpha: {alpha_max:.3e},\
        #         Min Alpha: {alpha_min:.3e}, Max Alpha_old: {alpha_old_max:.3e}, Min Alpha_old: {alpha_old_min:.3e}")
        print("----------------------")
        print(f"Iteration: {iteration:3d}, Error_L2           : {error_L2:3.4e}")
        print(f"Iteration: {iteration:3d}, Max Alpha          : {alpha_max:.3e}")
        print(f"Iteration: {iteration:3d}, Min Alpha          : {alpha_min:.3e}")
        print(f"Iteration: {iteration:3d}, Max Alpha_old      : {alpha_old_max:.3e}")
        print(f"Iteration: {iteration:3d}, Min Alpha_old      : {alpha_old_min:.3e}")

def alternate_minimization(state,problem_u,solver_alpha_snes,parameters=alt_min_parameters,monitor=None):
    u           = state["u"]
    alpha       = state["alpha"]
    alpha_old   = state["alpha_old"]
    # Set previous alpha as alpha_old
    alpha.vector.copy(alpha_old.vector)
    #for iteration in range(parameters["max_iter"]):  
    contador=0
    with open(savedir + "/monitor_alpha.txt", 'w') as archivo:
        for factor in np.linspace(2,1,21):
            altura.value=altura_0/factor
            
            for iteration in range(10):                         
                # solve displacement
                print("Solving Elasticity")
                problem_u.solve()
                print(f"Max_U[2]: {np.max(u[2].x.array):.5f}, Min_U[2]: {np.min(u[2].x.array):.5f}\n")
                # solve damage
                print("Solving Alpha")
                solver_alpha_snes.solve(None, alpha.vector)
                
                
                archivo.write(f"Factor: {factor:.10f}, Max_alpha: {np.max(alpha.x.array):.5e}, Min_alpha: {np.min(alpha.x.array):.5e}\n")
                print(f"Factor: {factor:.10f}, Max_alpha: {np.max(alpha.x.array):.5f}, Min_alpha: {np.min(alpha.x.array):.5f}\n")
                
                if np.min(alpha.x.array) < -1.0:
                    break
                # check error and update
                L2_error    = dolfinx.fem.form(ufl.inner(alpha - alpha_old, alpha - alpha_old) * dx)
                error_L2    = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error),op=MPI.SUM))
                
                # Monitor of solutions
                if monitor is not None:
                    monitor(state, iteration, error_L2)
                # check error                      
                if error_L2 <= parameters["atol"]:
                    break
                # uptade alpha_old with the value of alpha
                alpha.vector.copy(alpha_old.vector)

            guardar(u,alpha,contador,densidad_energy2)
            contador+=1
            if np.min(alpha.x.array) < -1.0:
                break
       
    return (error_L2, iteration)

# ================================================================================
# Let us define the total energy of the system as the 
# sum of elastic energy, dissipated energy due to the 
# damage and external work due to body forces. 
# ================================================================================
elastic_energy      = 0.5 * ufl.inner(sigma(u,alpha), eps(u)) * dx 
densidad_energy2 =   0.5*ufl.inner(Dev(sigma(u,alpha)),Dev(eps(u))) \
                        +(1-kappa)*ufl.inner(Sph(sigma(u,alpha)),Sph(eps(u)))
elastic_energy2 =   (densidad_energy2)*dx
#dissipated_energy   = Gc / float(c_w) * (w(alpha) / ell + ell * ufl.dot(ufl.grad(alpha), ufl.grad(alpha))) * dx
dissipated_energy   = (w(alpha)  + w(1)*ell**2 * ufl.dot(ufl.grad(alpha), ufl.grad(alpha)) )* dx

external_work       = ufl.dot(f, u) * dx

if presion_lateral: 
    bounbdary_energy      = ufl.dot(g_bc*n,u)*ds(1)+ufl.dot(g_bc*n,u)*ds(3)+ufl.dot(g_bc*n,u)*ds(4)\
                                +ufl.dot(g_bc*n,u)*ds(5)+ufl.dot(g_bc*n,u)*ds(6)
else:
    bounbdary_energy      = ufl.dot(g_bc*n,u)*ds(1)
total_energy        = elastic_energy + dissipated_energy - external_work - bounbdary_energy
#total_energy        = elastic_energy  - external_work - bounbdary_energy

total_energy2        = elastic_energy2 + dissipated_energy - external_work - bounbdary_energy


#total_disip         =  0.5*kappa*ufl.inner(Sph( sigma_der(u,alpha)),Sph( eps(u)))*alpha_dot*dx


# ================================================================================
# Weak form of elasticity problem. This is the formal 
# expression for the tangent problem which gives us the 
# equilibrium equations
# ================================================================================
E_u             = ufl.derivative(total_energy,u,ufl.TestFunction(V_vec))
E_du            = ufl.replace(E_u,{u: ufl.TrialFunction(V_vec)})
E_alpha         = ufl.derivative(total_energy2,alpha,ufl.TestFunction(V_sca ))
E_alpha_alpha   = ufl.derivative(E_alpha,alpha,ufl.TrialFunction(V_sca ))
jacobian        = dolfinx.fem.form(E_alpha_alpha)
residual        = dolfinx.fem.form(E_alpha)
# Displacement problem
problem_u   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_du), L=ufl.rhs(E_du), bcs=bcs_u, u=u,
                                      petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-10"})
    
# Damage problem
damage_problem      = SNESProblem(residual, jacobian, alpha, bcs_alpha)
b                   = dolfinx.la.create_petsc_vector(V_sca.dofmap.index_map, V_sca.dofmap.index_map_bs)
J                   = dolfinx.fem.petsc.create_matrix(damage_problem.a)
solver_alpha_snes   = PETSc.SNES().create()
solver_alpha_snes.setType("vinewtonrsls")
solver_alpha_snes.setFunction(damage_problem.F, b)
solver_alpha_snes.setJacobian(damage_problem.J, J)
solver_alpha_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_alpha_snes.getKSP().setType("cg")
solver_alpha_snes.getKSP().setTolerances(rtol=1.0e-8)
solver_alpha_snes.getKSP().getPC().setType("jacobi")
# We set the bound (Note: they are passed as reference and not as values)
solver_alpha_snes.setVariableBounds(alpha_lb.vector,alpha_ub.vector)
#solver_alpha_snes.setVariableBounds(alpha_lb.vector,1.0e20*alpha_ub.vector)

# Set initial condition for damage
#with alpha.vector.localForm() as alpha_local:
#    alpha_local.set(0)
alpha.interpolate(initial_condition)

print(f"----------------------------")
print(f"-- Solving Cavity Problem --")
print(f"----------------------------")

alternate_minimization(state,problem_u,solver_alpha_snes,parameters=alt_min_parameters,monitor=simple_monitor)
# save solutions
#guardar(u,alpha,t,densidad_energy2)

u_file.close()
alpha_file.close()
sigma_file.close()
eps_file.close()
sigma_desviador_file.close()
sigma_vonmisses_file.close()
energia_alpha_file.close()



fDatos      =   open(savedir + '/Parametros.txt', 'w')
buff='E     =%1.1e\n'%(E.value);fDatos.write(buff) ; print (buff)
buff='nu    =%f\n'%(nu.value);fDatos.write(buff)     ; print (buff)
buff='ell   =%f\n'%(ell.value);fDatos.write(buff)   ; print (buff)
buff='C_1   =%s\n'%(C_1);fDatos.write(buff)     ; print(buff)
buff='C_2   =%s\n'%(C_2);fDatos.write(buff)     ; print(buff)
buff='C_3   =%f\n'%(C_3);fDatos.write(buff); print (buff)
fDatos.close()
