import math
import numpy as np
import statistics
import time
import random



class Vector:
    def __init__(self, co_ordinates):
        """
        :param co_ordinates: a tuple ot co_ordinates of the given vertex
        """
        self.co_ordinates = np.array(co_ordinates)

    def compute_magnitude(self):
        """
        :return: returns the magnitude of the given vertex
        """
        return np.linalg.norm(self.co_ordinates)

    def dot_product(self, b):
        return np.dot(self.co_ordinates, b.co_ordinates)

    def cross_product(self, b):
        return Vector(np.cross(self.co_ordinates, b.co_ordinates))

    def to_file(self):
        return 'v ' + ' '.join((str(j) for j in self.co_ordinates)) + '\n'

    def __str__(self):
        return " , ".join((str(j) for j in self.co_ordinates))

    def angle(self, v1):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                    >>> angle_between((1, 0, 0), (0, 1, 0))
                    1.5707963267948966
                    >>> angle_between((1, 0, 0), (1, 0, 0))
                    0.0
                    >>> angle_between((1, 0, 0), (-1, 0, 0))
                    3.141592653589793
            taken from: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
            """

        return #np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class DisplacementVector(Vector):
    def __init__(self, final, initial):
        diff = final.co_ordinates - initial.co_ordinates
        super().__init__(diff)

class Face:
    def __init__(self, vertices_index):
        self.n  = len(vertices_index)
        self.vertices_index = np.array(vertices_index)

    def surface_normal(self, list_of_vertices):
        #assuming faces are co-planer

        normals = []
        for i in range(1, 2, 2):
            vector1 = list_of_vertices[self.vertices_index[i - 1]]
            if i + 1 == self.n:
                vector2 = list_of_vertices[self.vertices_index[0]]
            else:
                vector2 = list_of_vertices[self.vertices_index[i + 1]]
            edge1 = DisplacementVector(vector1, list_of_vertices[self.vertices_index[i]])
            edge2 = DisplacementVector(vector2, list_of_vertices[self.vertices_index[i]])
            normals = (edge1.cross_product(edge2))
        return normals

    def to_file(self):
        return 'f ' + ' '.join((str(j+1) for j in self.vertices_index)) + '\n'

    def __str__(self):
        return 'f ' + ' '.join((str(j+1) for j in self.vertices_index))

    def change_orientation(self):
        """
        reverse the order of the vertices
        """

        self.vertices_index = np.flipud(self.vertices_index[::])


    def has_vertex(self, i):
        return i in self.vertices_index

    def get_edges(self):
        edges = set()
        for i in range(self.n):
            start = self.vertices_index[i]
            end = self.vertices_index[(i + 1) % self.n]
            edges.add((start, end))

        return edges


    def area_of_face(self, list_of_vertices):
        if self.n == 3:
            return (0.5) * self.surface_normal(list_of_vertices).compute_magnitude()
        else:
            return self.surface_normal(list_of_vertices).compute_magnitude()


class Mesh:
    def __init__(self, filename):
        """
        :param filename: the name of the text file to open and read
        """
        txtFile = open(filename)
        self.file_name = filename.replace('.txt','')
        self.vertices, self.faces = (int(i) for i in txtFile.readline().strip().split())
        self.list_of_vertices = [] #
        self.list_of_faces = []
        self.directed_edges = dict()
        for i in range(self.vertices):
            line = txtFile.readline()
            vertices = tuple(float(i) for i in line.strip().split())
            self.list_of_vertices.append(Vector(vertices))
        self.dimensions = len(vertices)
        self.edges = dict()
        for i in range(self.faces):
            line = txtFile.readline()
            faces = tuple(int(k) for k in line.strip().split())
            self.list_of_faces.append(Face(faces))
        self.simplify_vertices()
        self.update_edges()


    def simplify_vertices(self):
        points_vertices = dict()
        for i in range(self.vertices):
            current_verex = self.list_of_vertices[i]
            co_ord = tuple(current_verex.co_ordinates)
            if co_ord in points_vertices:
                points_vertices[co_ord].append(i)
            else:
                points_vertices[co_ord] = [i]
        counter = 0
        for i in points_vertices:
            points_vertices[i] = counter
            counter += 1

        for i in range(self.faces):
            current_face =  self.list_of_faces[i]
            vertices_indices = current_face.vertices_index
            co_ordinates = [tuple(self.list_of_vertices[i].co_ordinates) for i in vertices_indices]
            new_indices = [points_vertices[i] for i in co_ordinates]
            self.list_of_faces[i].vertices_index = np.array(new_indices)

        self.list_of_vertices = [None for i in range(len(points_vertices))]
        for i , j in enumerate(points_vertices):
            self.list_of_vertices[i] = Vector(j)

        self.vertices = len(self.list_of_vertices)




    def update_edges(self):
        self.directed_edges = {}
        for i in range(self.faces):
            faces = tuple(int(k) for k in self.list_of_faces[i].vertices_index)
            for j in range(len(faces)):
                start = faces[j]
                if j == (len(faces) - 1):
                    end = faces[0]
                else:
                    end = faces[j + 1]
                if (start, end) in self.directed_edges:
                    self.directed_edges[(start, end)].append(i)
                else:
                    self.directed_edges[(start, end)] = [i]


    def list_of_unoriented(self):
        """
        returns the a list of unoriented faces. It assumes that the first face is correctly oriented,
        and uses that as a referrance for the the other vector. It checks if the neighboring faces have the same directed edge,
        if yes, the neighboring edge is differently oriented, and hence we put that face into inward oriented set.
        It computes both inward, and outward(assuming that the first face is outward)
        pointing faces, and returns the list who have minimum length.

        This assumes that the mesh, is connected.
        """
        visited_faces = set()
        edges = set()
        to_visit = list([10])
        self.un_oriented_egdes = set()
        outward = set()
        inward = set()
        while to_visit:
            """
            using a BFS type approach to navigate through all the faces.
            """
            current_face_index = to_visit.pop(0)
            if current_face_index in visited_faces:
                continue
            current_face = self.list_of_faces[current_face_index]
            visited_faces.add(current_face_index)
            next_edges = current_face.get_edges()
            outward_face = True

            for i in next_edges:
                list_of_faces = self.directed_edges.get(i, []) + self.directed_edges.get(i[::-1], [])
                for j in list_of_faces:
                    if j not in visited_faces and j != current_face_index:
                        to_visit.append(j)

                if i in edges: #this implies that the face has incorrect orientation
                    outward_face = False
                    self.un_oriented_egdes.add(i)

            if not outward_face:
                inward.add(current_face_index)
            else:
                edges.update(next_edges)
                outward.add(current_face_index)

        if len(inward) < len(outward):
            return inward
        else:
            return outward

    def vertices_dictionary(self):
        """
        returns a dictionary in the form of vertex: list of faces connected to that vertex.
        """
        vertices_dictionary = {}
        for i in range(self.vertices):
            vertices_dictionary[i] = []
            for j in range(self.faces):
                current_face = self.list_of_faces[j]
                if current_face.has_vertex(i):
                    vertices_dictionary[i].append(j)
        return vertices_dictionary

    def check_manifold_edges(self):
        """
        returns True if all edges are connected to exactly 2 triangles else returns false.
        """
        ed = set()

        for i in self.directed_edges:
            if i[::-1] not in self.directed_edges or len(self.directed_edges[i]) > 1:
                ed.add(i)
        return len(ed)


    def convert_to_obj(self, filename = ''):
        """
        :param filename: the fileanme to output the current mesh to.
        """
        if not filename:
            filename = self.file_name + ".obj"
        newfile = open(filename, 'w')
        for i in range(self.vertices):
            to_write = self.list_of_vertices[i].to_file()
            newfile.write(to_write)
        for i in range(self.faces):
            to_write = self.list_of_faces[i].to_file()
            newfile.write(to_write)
        return True

    def normals(self):
        """
        returns a list of normals of all the faces of the mesh.
        """
        normals = []
        for i in range(self.faces):
            current_face = self.list_of_faces[i]
            normals.append(current_face.surface_normal(self.list_of_vertices))
        return normals

    def correct_orientation(self):
        """
        corrects the orientation of the given indices of faces, utilizing the un_oriented_faces specified by the
        function list_of_unoriented.

        """
        un_oriented_faces = self.list_of_unoriented()
        for i in un_oriented_faces:
            face = self.list_of_faces[i]
            face.change_orientation()
        self.update_edges()


    def is_manifold(self):
        if not self.check_manifold_edges():
            return True
        return False

    def compute_area(self):
        return sum(face.area_of_face(self.list_of_vertices) for face in self.list_of_faces)


"""
m = Mesh('testMeshes/validTriangleMesh.txt')
print(m.compute_area())
print(m.list_of_unoriented4())

m = Mesh('testMeshes/validQuadrilateralMesh.txt')
print(m.list_of_unoriented4())

m = Mesh('testMeshes/validTriQuadMesh.txt')
print(m.list_of_unoriented4())

m = Mesh('testMeshes/geometry1.txt')
print(m.list_of_unoriented4())


m = Mesh('testMeshes/geometry6.txt')
print(m.list_of_unoriented4())


m = Mesh('testMeshes/geometry5.txt')
print(m.compute_area())
#m.convert_to_obj(('test_before.obj'))
print(sorted(m.list_of_unoriented()))
print(m.correct_orientation())
print(m.list_of_unoriented())
#m.check_manifold_edges()

m = Mesh('testMeshes/geometry4.txt')
print(m.list_of_unoriented4())

m = Mesh('testMeshes/geometry5.txt')
print(m.list_of_unoriented4())

m = Mesh('testMeshes/geometry6.txt')
print(m.list_of_unoriented4())

print(len(m.list_of_unoriented()))
print(m.compute_area())
m.correct_orientation(m.list_of_unoriented())
m.convert_to_obj('geometry5_conv.obj')
print(len(m.list_of_unoriented()))
print('DONE')
"""
