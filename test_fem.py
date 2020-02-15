import numpy as np
from fem import *
from scipy.integrate import quadrature, dblquad
import matplotlib.pyplot as plt


func = lambda x, y: (x**2) + (y**2)
#func = lambda x, y: (x**3) + (y**3)

del_func = lambda x,y: -4
#del_func = lambda x,y: -(6*x + 6*y )

mesh4 = "../meshFiles/unit_square.txt"
square = FiniteElementSolver(mesh4, del_func)
mesh5 = "../meshFiles/square_area_2.txt"
square_2 = FiniteElementSolver(mesh5, del_func)



mesh1 = "../meshFiles/basicMesh.txt"
f1 = FiniteElementSolver(mesh1, del_func)


mesh2 = "../meshFiles/basicMesh1.txt"
f2 = FiniteElementSolver(mesh2, del_func)


sq_mesh = "../meshFiles/squarePlate0.txt"
sq = FiniteElementSolver(sq_mesh, del_func)

mesh3 = "../meshFiles/testMesh2.txt"
f3 = FiniteElementSolver(mesh3, del_func)


mesh4 = "../meshFiles/testMesh2.txt"
f4 = FiniteElementSolver(mesh4, del_func)


mesh4 = "../meshFiles/circle_field.txt"
c = FiniteElementSolver(mesh4, del_func)




# Pytest requires its test functions to be prefixed with 'test_xxx' as shown below. https://docs.pytest.org/en/latest/.
# Helper/private functions can be named in whatever way you like



def test_area_ratio(): #part e
    mesh4 = "../meshFiles/unit_square.txt"
    square = FiniteElementSolver(mesh4)
    mesh5 = "../meshFiles/square_area_2.txt"
    square_2 = FiniteElementSolver(mesh5)
    assert np.linalg.det(square.jacobian(element_num = 0, local = (0,0))) == 1 #each side has length 1
    assert np.linalg.det(square.jacobian(element_num=1, local=(0, 0))) == -1  # each side has length 1
    assert np.linalg.det(square.jacobian(element_num=0, local=(0.5, 0.5))) == 1  # each side has length 1
    assert np.linalg.det(square_2.jacobian(element_num=0, local=(0, 0))) == 4 #each side has length 2, hence area = 2^2
    assert np.linalg.det(square_2.jacobian(element_num=1, local=(0, 0))) == 0  # each on same length



def test_local_to_global_enum(): #1(c)
    mesh1 = "../meshFiles/basicMesh.txt"
    f1 = FiniteElementSolver(mesh1)
    assert f1.local_to_global_enum(0, 0) == 8 #testing for individual node nums
    assert f1.local_to_global_enum(3, 3) == 8 #testing for individual node nums
    assert f1.local_to_global_enum(1, 2) == 7 #testing for individual node nums



def test_co_ordinate_conversion(): #1d
    mesh1 = "../meshFiles/basicMesh.txt"
    f1 = FiniteElementSolver(mesh1)
    mesh5 = "../meshFiles/square_area_2.txt"
    square_2 = FiniteElementSolver(mesh5)
    assert np.array_equal(f1.vertices[8], (f1.co_ordinate_conversion(element_num = 0, local = (0, 0))))
    assert np.array_equal(f1.vertices[6], (f1.co_ordinate_conversion(element_num = 0, local=(1, 0))))
    assert np.array_equal(f1.vertices[3], (f1.co_ordinate_conversion(element_num = 0, local=(1, 1))))
    assert np.array_equal(f1.vertices[7], (f1.co_ordinate_conversion(element_num = 0, local=(0, 1))))
    assert np.array_equal(square_2.vertices[5], (square_2.co_ordinate_conversion(element_num = 1, local=(0, 1))))


def test_derivative_conversion(): #1 d
    """
    computed on paper to test the answers
    """
    mesh1 = "../meshFiles/basicMesh.txt"
    f1 = FiniteElementSolver(mesh1)
    mesh4 = "../meshFiles/unit_square.txt"
    square = FiniteElementSolver(mesh4)
    #print(f1.derivative_conversion(0, (0.0, 0.0)))
    #print(square.derivative_conversion(0, (0.0, 0.0)))
    #print(square.derivative_conversion(0, (0.5, 0.5)))
    #print(square.derivative_conversion(0, (0.9, 0.9)))

	#return 0



def test_element_stiffness(): # 1(g)
    """computed on paper to verify"""
    mesh4 = "../meshFiles/testMesh2.txt"
    f4 = FiniteElementSolver(mesh4)
    assert 4.75 <= f4.element_stiffness(0)[0, 0] * 5  <  4.77 #should be 4.76, example from page 63 of numerical methods notes.

def test_element_RHS(): # 1(h)
    mesh4 = "../meshFiles/testMesh2.txt"
    f4 = FiniteElementSolver(mesh4, lambda x,y: 0)
    assert np.array_equal(np.zeros(4), f4.element_rhs(0)) #since charge density is zero

def test_global_stiffness(): # 1(i)
    mesh1 = "../meshFiles/basicMesh.txt"
    f1 = FiniteElementSolver(mesh1)
    mesh2 = "../meshFiles/basicMesh1.txt"
    f2 = FiniteElementSolver(mesh2)
    f1.construct_global_stiffness() #constructs and returns global stiffness matrix
    f2.construct_global_stiffness() #constructs and returns global stiffness matrix


def test_global_RHS(): # 1(j)
    mesh1 = "../meshFiles/basicMesh.txt"
    f1 = FiniteElementSolver(mesh1, lambda x,y: 0)
    f1.setup_system()
    mesh2 = "../meshFiles/basicMesh1.txt"
    f2 = FiniteElementSolver(mesh2, lambda x,y: 0)
    f2.setup_system()
    assert np.array_equal(np.zeros(len(f1.vertices)), f1.RHS) #constructs and returns RHS vector
    assert np.array_equal(np.zeros(len(f2.vertices)), f2.RHS) #constructs and returns RHS vector

def test_apply_dirichlet(): # 1(k)
    mesh1 = "../meshFiles/basicMesh.txt"
    f1 = FiniteElementSolver(mesh1, lambda x, y: x*y)
    f1.setup_system() #setting up stiffness and RHS, global.
    f1.apply_boundary_conditions([0, 2, 3], [0, 5, -5])
    assert f1.global_stiffness[0,0] == 1 and np.sum(f1.global_stiffness[0,:]) == 1 and f1.RHS[0] == 0
    assert f1.global_stiffness[2, 2] == 1 and np.sum(f1.global_stiffness[2, :]) == 1 and f1.RHS[
        2] == 5
    assert f1.global_stiffness[3, 3] == 1 and np.sum(f1.global_stiffness[3, :]) == 1 and f1.RHS[
        3] == -5


def test_system():
    """will produce multiple plots for mulitple meshes"""
    functions = [lambda x,y: x + y, lambda x,y: 2*(x**2) + 3*(y**2), lambda x,y: (x**3) + (y**3), lambda x,y: 3*(x**3) - 4*(y**2), lambda x, y: np.sin(math.pi * x) * np.sin(math.pi * y)]
    del_funcs = [lambda x,y: 0, lambda x,y: -10, lambda x,y: -(6*x  + 6*y), lambda x,y: -(18*x - 8), lambda x, y: 2 * (math.pi ** 2) * np.sin(math.pi * x) * np.sin(math.pi * y)]
    files = ["../meshFiles/basicMesh1.txt", "../meshFiles/basicMesh2.txt"]
    for i in range(len(functions)):
        func = functions[i]
        laplace = del_funcs[i]
        error_norms = []
        for mesh in files:
            errors = []
            solver = FiniteElementSolver(mesh, f = laplace)
            solver.setup_system()
            solver.automatic_BC(func)
            solution = solver.solve_system(with_plot = True)
            for vertex_num in range(len(solver.vertices)):
                if solution[vertex_num] != func(*solver.vertices[i][:2]):
                    errors.append((solution[vertex_num] - func(*solver.vertices[vertex_num][:2])) ** 2)
            error_norms.append(np.linalg.norm(errors))

        for i in range(len(error_norms) - 1):

            """since meshes are arrangeed in the increasing order of subdivision, we expect the error to reduce with increasing subdivision"""
            assert round(error_norms[i], 10) >= round(error_norms[i + 1], 10)


def test_jacobi_iterative(): # 1(m)
    N = 35 # or any greater value runs for all test cases below with minimum error
    A1 = [[4.0, -2.0, 1.0], [1.0, -3.0, 2.0], [-1.0, 2.0, 6.0]]
    b1 = [1.0, 2.0, 3.0]
    x1 = [1.0, 1.0, 1.0]
    from_jacobi = jacobi_iterative(A1, b1, N, x1)
    from_numpy = np.linalg.solve(A1, b1)
    assert np.allclose(from_jacobi,from_numpy) == True

    from_jacobi = jacobi_iterative(A1, b1, N)
    from_numpy = np.linalg.solve(A1, b1)
    assert np.allclose(from_jacobi, from_numpy) == True

    A2 = np.array([[2.0, 1.0], [5.0, 7.0]])
    b2 = [11.0, 13.0]
    x2 = [1.0, 1.0]
    from_jacobi = jacobi_iterative(A2, b2, N,x2)
    from_numpy = np.linalg.solve(A2, b2)
    assert np.allclose(from_jacobi, from_numpy) == True

    GL = 1.6
    d = 0.8
    A3 = np.array([
        [1.0, 0, 0, 0, 0],
        [GL, -(d + 1), 1.0, 0, 0],
        [0, d, -(d + 1), 1.0, 0],
        [0, 0, d, -(d + 1), 1.0],
        [0, 0, 0, 0, 1.0]])
    b3 = [0.5, 0, 0, 0, 0.1]
    from_jacobi = jacobi_iterative(A3, b3, N)
    from_numpy = np.linalg.solve(A3, b3)
    assert np.allclose(from_jacobi, from_numpy) == True

    #testing with mesh

    mesh2 = "../meshFiles/basicMesh1.txt"
    f2 = FiniteElementSolver(mesh2, lambda x, y: 0)
    f2.setup_system()
    f2.automatic_BC(lambda x,y: x + y)
    f2.solve_system(with_plot = True) #without jacobi
    f2.solve_system(with_plot=True, jacobi= True) #with jacobi


def test_end_to_end(): # 1(n)
    mesh_name = "../meshFiles/basicMesh2.txt"
    func = lambda x, y: np.sin(math.pi * x) * np.sin(math.pi * y)
    end_to_end(mesh_name, charge_density=lambda x, y: -2 * (math.pi ** 2) * np.sin(math.pi * x) * np.sin(math.pi * y),
               permivitity_of_medium=1, boundary_vals=[],
               boundary_vertices=[],
               anaytical_func=func)

    mesh_name = "../meshFiles/basicMesh2.txt"
    func = lambda x, y: (x ** 2) + (y ** 2)
    end_to_end(mesh_name, charge_density = lambda x,y: 4, permivitity_of_medium = 1, boundary_vals= [], boundary_vertices=[], anaytical_func =func) #passing the analytical function so that it computes the BC automatically

    mesh_name = "../meshFiles/squarePlate0.txt"
    func = lambda x, y: np.sin(math.pi * x) * np.sin(math.pi * y)
    end_to_end(mesh_name, charge_density=lambda x, y: -2 * (math.pi ** 2) * np.sin(math.pi * x) * np.sin(math.pi * y), permivitity_of_medium=1, boundary_vals=[],
               boundary_vertices=[],
               anaytical_func=func)  # passing the analytical function so that it computes the BC automatically


def test_gaussian_integral(): # 1(f)
    """1D test cases"""
    oneDfunction = lambda x: 5 * x ** 2 + 2
    circle_trig = lambda thetha: (np.sin(thetha)) ** 2 + (np.cos(thetha)) ** 2

    result_oneDfunction_scipy = round(quadrature(oneDfunction, 0.25,0.75)[0], 3)
    result_oneDfunction = round(gaussian_integral(oneDfunction, 0.25, 0.75, 1), 3)

    assert (result_oneDfunction == result_oneDfunction_scipy)

    result_circle_trig_scipy = round(quadrature(circle_trig, np.pi/4, np.pi/2)[0], 3)
    result_circle_trig =  round(gaussian_integral(circle_trig, np.pi/4, np.pi/2, 1), 3)
    assert (result_circle_trig == result_circle_trig_scipy)

    result_cos_scipy = round(quadrature(np.cos, 0, np.pi/4)[0], 3)
    result_cos = round(gaussian_integral(np.cos, 0, np.pi/4, 1), 3)
    assert (result_cos == result_cos_scipy)

    """2D test cases"""
    circle = lambda x, y: (x ** 2) + (y ** 2)
    twoDfunction = lambda y, x: x * y ** 2

    result_circle_scipy = round(dblquad(circle, 0, 1, 0, 1)[0], 3)
    result_circle = round(gaussian_integral(circle, 0, 1, 2), 3)
    assert result_circle == result_circle_scipy

    result_twoDfunction_scipy = round(dblquad(twoDfunction, 0, 1, 0, 1)[0], 3)
    result_twoDfunction = round(gaussian_integral(twoDfunction, 0, 1, 2), 3)
    assert result_twoDfunction == result_twoDfunction_scipy


# test_area_ratio()
# test_local_to_global_enum()
# test_co_ordinate_conversion()
# test_derivative_conversion()
# test_element_stiffness()
# test_element_RHS()
# test_global_stiffness()
# test_global_RHS()
# test_apply_dirichlet()
# test_system()
# test_jacobi_iterative()
# test_end_to_end()
# test_gaussian_integral()

