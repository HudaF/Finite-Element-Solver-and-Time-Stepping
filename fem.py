from MeshLib import *
import numpy as np
import math
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class FiniteElementSolver:
    def __init__(self, filename, f = 0):
        """
        :param filename: the underlying mesh file.
        :param f, the function on the right side of the PDE.
        """
        mesh = Mesh(filename)
        self.length_of_element = 4
        self.elements = np.zeros((mesh.faces, self.length_of_element), dtype = np.int64) #as element has 4 faces.
        self.vertices = [i.co_ordinates for i in mesh.list_of_vertices]
        for mesh_num in range(mesh.faces):
            assert len(mesh.list_of_faces[mesh_num].vertices_index) == self.length_of_element #quadilateral
            self.elements[mesh_num] = np.array(mesh.list_of_faces[mesh_num].vertices_index)

        self.basis_co_effients = self.derive_co_efficients()
        self.basis_dt, self.basis_ds = self.derive_co_efficients_derivative()
        self.charge_density = f

    def setup_system(self):
        """
        sets up the global stifness matrix, and load vector.
        MUST be called before the application of boundary conditions and solve_system.
        """
        if self.charge_density != 0:
            self.global_stiffness = self.construct_global_stiffness()
            self.RHS = self.construct_global_rhs()

    def derive_co_efficients(self, degree = 1):
        """
        sets the up the equation a + b*(t) + c * (s) + d * (s) * (t) such that at (0,0) only point 1 is active, at
        (1, 0) only point 2 is active, at point (1, 1) only point 3 is active, and at (1,0) only point 4 is active.
        """
        assert  degree == 1 #defining only for degree 1
        A = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]]
        b = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        co_effs = np.linalg.solve(A,b)

        return co_effs

    def derive_co_efficients_derivative(self):
        """
        function to compute the co-efficients of Ni/dt and Ni/ds.
        if Ni = a + b*(t) + c*(s) + d * (s) * (t)
        Ni/dt = b + d * (s)
        Ni/ds = c + d * (t)
        :return:
        """
        constant_differential = 0,0
        co_effs_t = self.basis_co_effients.copy()
        co_effs_t[0], co_effs_t[2] = constant_differential #as terms having no t would become 0
        co_effs_s = self.basis_co_effients.copy()
        co_effs_s[0], co_effs_s[1] = constant_differential  # as terms having no s would become 0
        return co_effs_t, co_effs_s

    def derivative_at_t(self, i, t, s):
        """
        if Ni = a + b*(t) + c*(s) + d * (s) * (t)
        Ni/dt = b + d * (s)
        """
        return self.basis_dt[1][i] + (self.basis_dt[3][i] * s)

    def derivative_at_s(self, i, t, s):
        """
        if Ni = a + b*(t) + c*(s) + d * (s) * (t)
        Ni/ds = c + d * (t)
        """
        return self.basis_ds[2][i] + (self.basis_ds[3][i] * t)

    def basis_function(self, N, t, s):
        #following the equation a + b*(t) + c * (s) + d * (s) * (t)
        return self.basis_co_effients[0][N] + (self.basis_co_effients[1][N] *t) + (self.basis_co_effients[2][N] * s) + (self.basis_co_effients[3][N] * t * s)

    def jacobian(self, element_num, local):
        """
        slide 33 for reference.
        :param element_num: the element we are dealing with
        :param local: tuple -> (t,s) representing the parametric co_ordinates
        :return: returns a jacobian
        """
        j = np.zeros((2,2))
        t, s = local

        #dx/dt
        j[0, 0] = np.sum([self.derivative_at_t(i, t, s) * self.vertices[self.local_to_global_enum(element_num, i)][0] for i in range(self.length_of_element)])
        #dy/dt
        j[0, 1] = np.sum([self.derivative_at_t(i, t, s) * self.vertices[self.local_to_global_enum(element_num, i)][1] for i in range(self.length_of_element)])
        #dx/ds
        j[1, 0] = np.sum([self.derivative_at_s(i, t, s) * self.vertices[self.local_to_global_enum(element_num, i)][0] for i in range(self.length_of_element)])
        #dy/ds
        j[1, 1] = np.sum([self.derivative_at_s(i, t, s) * self.vertices[self.local_to_global_enum(element_num, i)][1] for i in range(self.length_of_element)])

        return j

    def derivative_conversion(self, element_num, local):
        """
        notes page 49 for referennce.
        :param element_num: the element we are dealing with
        :param local: tuple -> (t,s) representing the parametric co_ordinates
        :return: a tuple representing d(Ni)/dx and d(Ni)/dy
        """

        jacobian = self.jacobian(element_num, local)
        assert (np.linalg.det(jacobian) != 0), "Jacobian non-invertible"
        jacobian_inv = np.linalg.inv(jacobian)

        B = np.zeros((2, self.length_of_element)) #array of derivates.
        t, s = local
        for i in range(self.length_of_element):
            B[:, i] = np.matmul(jacobian_inv, [self.derivative_at_t(i, t, s), self.derivative_at_s(i, t, s)])
        return B

    def element_rhs(self, element_num):
        def function_to_integerate(t, s):
            local = t,s
            global_co_ord = self.co_ordinate_conversion(element_num, local)
            basis_function_values = np.array([self.basis_function(i, t, s)for i in range(self.length_of_element)])
            f = self.charge_density(global_co_ord[0], global_co_ord[1])
            jacobian = np.linalg.det(self.jacobian(element_num, local))
            return f * basis_function_values * jacobian
        return gaussian_integral(function_to_integerate, lower_limit = 0, upper_limit= 1, dim = 2)

    def element_stiffness_2(self, element_num):
        element = np.zeros((self.length_of_element,self.length_of_element))
        a = 0.788675
        b = 0.21134286
        w = 0.25
        input = [(a, a), (a, b), (b, a), (b, b)]
        for i in range(self.length_of_element):
            for j in range(self.length_of_element):
                for ind in range(len(input)):
                    inp = input[ind]
                    B = self.derivative_conversion(element_num = element_num, local = inp)
                    element[i, j] += ((B[0, i] * B[0, j]) + (B[1, i] * B[1, j])) * w * np.linalg.det(self.jacobian(element_num=element_num, local = inp))
        return element



    def automatic_BC(self, func):
        """
        a function to test our FEM. func is the analytical function to apply the values at the boundary.
        We assume that the domain is rectangular.
        func must have the signature, func(x,y)
        """
        x_min = min([i[0] for i in self.vertices])
        x_max = max([i[0] for i in self.vertices])
        y_min = min([i[1] for i in self.vertices])
        y_max = max([i[1] for i in self.vertices])
        """getting the rectangular region"""
        boundary = []
        boundary_val = []
        for i in range(len(self.vertices)):
            if self.vertices[i][0] == x_min or self.vertices[i][0] == x_max or self.vertices[i][1] == y_min or self.vertices[i][1] == y_max: #at the boundary
                boundary.append(i)
                boundary_val.append(func(self.vertices[i][0], self.vertices[i][1]))
        self.apply_boundary_conditions(vertices=boundary, values=boundary_val) #applying the obtained conditions


    def element_stiffness(self, element_num):
        def function_to_integerate(t, s):
            local = t,s
            B = self.derivative_conversion(element_num, local)
            B_T = np.transpose(B)
            jacobian = np.linalg.det(self.jacobian(element_num, local))
            return np.matmul(B_T, B) * jacobian
        return gaussian_integral(function_to_integerate, lower_limit= 0, upper_limit= 1, dim = 2)

    def construct_global_stiffness(self):
        E = 0.0005
        global_stiff = np.zeros((len(self.vertices), len(self.vertices)))
        for element_num in range(self.elements.shape[0]):
            #print(self.elements.shape[0], "E shape")
            element_stiffness = self.element_stiffness(element_num) #constructing the stiff for that one element
            for row in range(self.length_of_element): #adding the element stiff to global stiffness
                for col in range(self.length_of_element):
                    global_stiff[self.local_to_global_enum(element_num, row),
                                 self.local_to_global_enum(element_num, col)] += element_stiffness[row, col]

        assert(check_symmetric(global_stiff))
        return global_stiff

    def construct_global_rhs(self):
        global_rhs = np.zeros(len(self.vertices))
        for element_num in range(self.elements.shape[0]):
            element_rhs = self.element_rhs(element_num)  # constructing the RHS for one element
            for row in range(self.length_of_element):
                global_rhs[self.local_to_global_enum(element_num, row)] += element_rhs[row]
        return global_rhs



    def local_to_global_enum(self, element_num, local_freedom_num):
        """
        for part 1(c)
        :param element_num: the number of element
        :param local_freedom_num: the local number of node within that element
        :return: the global number of node
        """
        assert local_freedom_num < self.length_of_element
        return self.elements[element_num, local_freedom_num]

    def plot_potential(self):
        # potential_at_x_y = []
        # div = 100
        x = []
        y = []
        potential = []
        for i in range(len(self.vertices)):
            x.append(self.vertices[i][0])
            y.append(self.vertices[i][1])
            potential.append(self.solution[i])
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(potential)
        plt.colorbar(m, label='scalar field q')
        plt.scatter(x = x,  y = y, c = potential)
        plt.legend()
        plt.show()





    def co_ordinate_conversion(self, element_num, local):
        """
        :param element_num: int -> the number of the element
        :param local: tuple (t, s), representing the local co_ordinate
        :return: tuple(x,y,z) represting (t,s) in global frame of reference.
        """
        t = local[0]
        s = local[1]
        s_t_range = [-1,1]
        assert s_t_range[0] <= t <= s_t_range[1] and s_t_range[0] <= s <= s_t_range[1] #within bounds
        point = np.zeros(len(self.vertices[0])) #to store the global point
        for i in range(self.length_of_element):
            basis_func = self.basis_function(i, t, s)
            point += basis_func * self.vertices[self.elements[element_num, i]]
        return point


    def plot_element(self, element_num):
        """
        plots the element on x,y place.
        """
        for i in range(self.length_of_element):
            glob = self.local_to_global_enum(element_num, i)
            glob_1 = self.local_to_global_enum(element_num, (i + 1)% self.length_of_element)
            plt.plot([self.vertices[glob][0], self.vertices[glob_1][0]], [self.vertices[glob][1], self.vertices[glob_1][1]])
        plt.show()
    def apply_boundary_conditions(self, vertices, values):
        '''
        :param vertices: the list of vertices where we have the boundary conditions defined.
        :param values: the values at the boundary conditiions
        modifies the global stiffness matrix inplace.
        '''
        E = 0.005
        assert len(vertices) == len(values)
        for i in range(len(vertices)):
            self.global_stiffness[vertices[i],] = 0
            self.global_stiffness[vertices[i], vertices[i]] = 1
            self.RHS[vertices[i]] = values[i]

        assert np.linalg.det(self.global_stiffness) > E or np.linalg.det(self.global_stiffness) < -(E) #to insure the system become solvable.


    def solve_system(self, with_plot = False, jacobi = False):
        if not jacobi:
            self.solution = np.linalg.solve(self.global_stiffness, self.RHS) #solving Ku = b
        else:
            self.solution = jacobi_iterative(self.global_stiffness, self.RHS, N = 100)
        if with_plot:
            x = [i[0] for i in self.vertices]
            y = [i[1] for i in self.vertices]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(x,y, self.solution, shade = True, cmap=cm.Blues)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("Potential(x,y)")
            plt.show()
        return self.solution

def is_pos_def(x): #taken from stackoverflow
    return np.all(np.linalg.eigvals(x) > 0)


def check_symmetric(a, tol=1e-8):  #taken from stackoverflow
    return np.all(np.abs(a-a.T) < tol)


def gaussian_integral(func, lower_limit, upper_limit, dim = 2):
    '''
    :param func: the function to integeral. MUST follow this structure,
    if dim -> 1, the function must have this signature f(x).
    if dim -> 2, the function must have this signature f(x,y)
    :param lower_limit: the lower limit of the integral
    :param upper_limit: the upper limit of the integral
    :param dim: the dimensions of the function
    :return: a float, the result of the gauss integral.
    '''
    if dim == 1:
        N = 2
        input = [-1/math.sqrt(3), 1/math.sqrt(3)]
    if dim == 2:
        N = 4
        input = [(-1 / math.sqrt(3),-1 / math.sqrt(3)), (1 / math.sqrt(3),-1 / math.sqrt(3)), (1 / math.sqrt(3),1 / math.sqrt(3)), (-1 / math.sqrt(3),1/math.sqrt(3))]
    weights = [1 for i in range(N)]
    input = [np.array(i) for i in input] #because need np array functions to add
    out = 0
    for i in range(N):
        # conversion taken from https://www.sciencedirect.com/topics/engineering/gauss-quadrature
        x = ((upper_limit + lower_limit)/2) + (upper_limit - lower_limit)*input[i]/2
        #x = input[i]
        if dim == 1:
            out += (upper_limit - lower_limit) * (weights[i]) * func(x) / 2
        else:

            out += (((upper_limit - lower_limit)/2)**2) * (weights[i]) * func(x[0], x[1])
    return out

def end_to_end(mesh_file, charge_density, permivitity_of_medium, boundary_vertices, boundary_vals, anaytical_func = 0):
    """
    mesh_file -> name of the mesh file
    charge_density -> a python (or lambda) function which has the signature f(x,y).
    permivitity_of_medium -> a constant (int or float)
    boundary_vertices -> vertices which are at the boundary of the domain
    boundary_vals -> value of the given function at boundary.
    """
    rho_by_epsilon = lambda x,y: - 1 * charge_density(x,y)/permivitity_of_medium
    solver = FiniteElementSolver(mesh_file, rho_by_epsilon)
    solver.setup_system() #setting up stiffness and RHS
    if not boundary_vertices and not boundary_vals:
        solver.automatic_BC(anaytical_func)
    else:
        solver.apply_boundary_conditions(boundary_vertices, boundary_vals)
    solver.solve_system(with_plot = True) #solves the given system and plots in on xyz co-ordinates.



def jacobi_iterative(A, b, N, x=None):
    """
    Ax = b makes the system of linear equations.
    :param N: maximum number of iterations
    :return: x, the estimated solution
    """
    if x is None: # initial guess
        x = np.zeros(len(A[0]))

    A = np.array(A)
    D = np.diag(A) # Diagonal component
    R = A - np.diagflat(D) # Remainder component
    for i in range(N):
        x = (b - np.dot(R,x))/ D
    return x


# mesh1 = "../meshFiles/basicMesh.txt"
# check = FiniteElementSolver(mesh1)
# check.setup_system()
# print(check.derivative_conversion(0,(0.78865,0.21135)))
# print(check.derivative_conversion(0,(0.21135,0.78865)))
# print(check.derivative_conversion(0,(0.78865,0.78865)))
# print(check.derivative_conversion(0,(0.21135,0.21135)))
#
# B = check.derivative_conversion(0,(0.78865,0.21135))
# B_transpose = B.transpose()
# a = np.matmul(B_transpose,B)
#
# B = check.derivative_conversion(0,(0.21135,0.78865))
# B_transpose = B.transpose()
# b = np.matmul(B_transpose,B)
#
# B = check.derivative_conversion(0,(0.78865,0.78865))
# B_transpose = B.transpose()
# c = np.matmul(B_transpose,B)
#
# B = check.derivative_conversion(0,(0.21135,0.21135))
# B_transpose = B.transpose()
# d = np.matmul(B_transpose,B)
#
# e = a+b+c+d
# K = -0.25*e
# print(K)


# print(check.derivative_conversion(1,(0.21135,0.78865)))
# print(check.derivative_conversion(1,(0.78865,0.78865)))
# print(check.derivative_conversion(1,(0.21135,0.21135)))

# print ('circle trig',result_circle_trig)
# print ('trig func',result_trig)
# print('circle', result_circle)
# print('1D ',result_1D)

# print("gloabl derv")
# print(check.local_to_global_derivative(2,(0,0)))
# print(check.local_to_global_derivative(2,(0,1)))
# print(check.local_to_global_derivative(2,(1,0)))
# print(check.local_to_global_derivative(2,(1,1)))
# print(check.local_to_global_derivative(3,(0,0)))
# print(check.local_to_global_derivative(3,(0,1)))
# print(check.local_to_global_derivative(3,(1,0)))
# print(check.local_to_global_derivative(3,(1,1)))





