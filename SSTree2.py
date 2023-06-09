import numpy as np
from scipy.spatial import distance
import pickle
import os
from queue import PriorityQueue
class SSnode:
    # Inicializa un nodo de SS-Tree
    def __init__(self, leaf=False, points=None, children=None, data=None, parent=None):
        self.leaf     = leaf
        self.points   = points if points is not None else []
        self.children = children if children is not None else []
        self.data     = data if data is not None else [] 
        self.centroid = np.mean([p for p in self.points], axis=0) if self.points else None
        self.radius   = self.compute_radius()
        self.parent   = parent

    # Calcula el radio del nodo como la máxima distancia entre el centroide y los puntos contenidos en el nodo
    def compute_radius(self):
        if self.leaf:
            if not self.points:
                return 0
            return max([distance.euclidean(i,self.centroid) for i in self.points])
        else:
            if not self.children:
                return 0
            return max([distance.euclidean(i.centroid,self.centroid)+i.radius for i in self.children])

    # Verifica si un punto dado está dentro del radio del nodo
    def intersects_point(self, point):
        return self.radius > distance.euclidean(point,self.centroid) 

    # Actualiza el envolvente delimitador del nodo recalculando el centroide y el radio
    def update_bounding_envelope(self):
        #Para nodo interno: Promedio del centroide de los hijos
        #Para hojas: Promedio de los hijos
        #Llamar a recarcular radio
        if self.leaf:
            self.centroid = np.mean([point for point in self.points])
        else:
            self.centroid = np.mean([i.centroid for i in self.children])
        self.compute_radius()

    # Encuentra y devuelve el hijo más cercano al punto objetivo
    # Se usa para entrar el nodo correto para insertar un nuevo punto
    def find_closest_child(self, target):
        return min(self.children, key=lambda x: distance.euclidean(x.centroid, target))
    # Divide el nodo en dos a lo largo del eje de máxima varianza
    # m
    def split(self, m):
        eje = self.direction_of_max_variance()
        values = [point[eje] for point in self.points]
        values.sort()

        split_index = self.find_split_index(m)
        values_left = values[:split_index]
        values_right = values[split_index:]

        newNode1 = SSnode(points=[i for i in self.points if i[eje] in values_left])
        newNode2 = SSnode(points=[i for i in self.points if i[eje] in values_right])

        return newNode1, newNode2

    # Encuentra el índice en el que dividir el nodo para minimizar la varianza total
    def find_split_index(self,m):
        num_points = len(self.points)
        num_splits = num_points - 2 * m + 1

        best_split_index = None
        best_variance_reduction = float('inf')

        for i in range(num_splits):
            values_subset = self.points[i : i + 2 * m]
            variance_reduction = self.min_variance_split(values_subset, m)

            if variance_reduction < best_variance_reduction:
                best_variance_reduction = variance_reduction
                best_split_index = i + m

        return best_split_index

    # Encuentra la división que minimiza la varianza total
    def min_variance_split(self, values, m):
        total_variance = np.var(values)
        left_variances = [np.var(values[: i + 1]) for i in range(m - 1)]
        right_variances = [np.var(values[i + 1 :]) for i in range(m - 1)]

        variances_sum = [left_variances[i] + right_variances[m - i - 2] for i in range(m - 1)]
        best_split_index = np.argmin(variances_sum)

        variance_reduction = total_variance - variances_sum[best_split_index]
        return variance_reduction

    # Encuentra el eje a lo largo del cual los puntos tienen la máxima varianza
    def direction_of_max_variance(self):
        # Calculate the variances along each dimension
        variances = np.var(self.points, axis=0)
        

        max_variance_index = np.argmax(variances)
        
        return max_variance_index

    # Obtiene los centroides de las entradas del nodo
    def get_entries_centroids(self):
        # Completar aqui!
        return [np.mean(c,axis=0) for c in self.points]
        
        

class SSTree:
    # Inicializa un SS-Tree
    def __init__(self, M=None, m=None, filename=None):
        if filename is None:
            self.M = M
            self.m = m
            if M is not None and m is not None:
                self.root = SSnode(leaf=True)
            else:
                self.root = None
        else:
            if os.path.exists(filename):
                loaded_tree = self.load(filename)
                self.M = loaded_tree.M
                self.m = loaded_tree.m
                self.root = loaded_tree.root
            else:
                print(f"'{filename}' no existe.")
                self.M = None
                self.m = None
                self.root = None

    # Inserta un punto en el árbol
    def insert(self, point, data=None):
        if self.root is None:
            self.root = SSnode(leaf=True,points=[point],data=[data])
        else:
            self._insert(self.root,point,data)

    def _insert(self, node, point, data):
        if node.leaf:
            node.points.append(point)
            node.data.append(data)
            if len(node.points) > self.M:
                newNode1, newNode2 = node.split(self.m)
                if node.parent is None:
                    self.root = SSnode(children=[newNode1, newNode2], parent=None)
                    newNode1.parent = self.root
                    newNode2.parent = self.root
                else:
                    parent = node.parent
                    parent.children.remove(node)
                    parent.children.extend([newNode1, newNode2])
                    newNode1.parent = parent
                    newNode2.parent = parent
                    if len(parent.children) > self.M:
                        self._split(parent)
        else:
            closest_child = node.find_closest_child(point)
            self._insert(closest_child, point, data)        

    # Busca un punto en el árbol y devuelve el nodo que lo contiene si existe
    def search(self, target):
        return self._search(self.root, target)

    # Función recursiva de ayuda para buscar un punto en el árbol
    def _search(self, node, target):
        if node.leaf:
            return node if target in node.points else None
        else:
            for child in node.children:
                if child.intersects_point(target):
                    result = self._search(child, target)
                    if result is not None:
                        return result
        return None
    


    # Depth-First K-Nearest Neighbor Algorithm
    def knn(self, q, k=3):
        pq = PriorityQueue()
        self._knn(self.root, q, k, pq)
        return [pq.get()[1] for _ in range(pq.qsize())][::-1]

    # Función recursiva de ayuda para el algoritmo KNN
    def _knn(self, node, q, k, pq):
        if node.leaf:
            for i, point in enumerate(node.points):
                dist = distance.euclidean(point, q)
                if pq.qsize() < k:
                    pq.put((-dist, node.data[i]))
                elif dist < -pq.queue[0][0]:
                    pq.get()
                    pq.put((-dist, node.data[i]))
        else:
            for child in node.children:
                if child.intersects_point(q):
                    self._knn(child, q, k, pq)
    # Guarda el árbol en un archivo
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Carga un árbol desde un archivo
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)