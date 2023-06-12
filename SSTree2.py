import numpy as np
from scipy.spatial import distance
import pickle
import os
from queue import PriorityQueue
class SSnode:
    def __init__(self, leaf=False, points=None, children=None, data=None, parent=None):
        self.leaf = leaf
        self.points = points if points is not None else []
        self.children = children if children is not None else []
        self.data = data if data is not None else [] 
        self.centroid = self.compute_centroid()
        self.radius = self.compute_radius()
        self.parent = parent

    def compute_radius(self):
        if self.points is None or self.centroid is None:
            return 0
        elif self.leaf:
            return max([distance.euclidean(i.flatten(), self.centroid.flatten()) for i in self.points])
        else:
            if not self.children:
                return 0
            return max([distance.euclidean(i.centroid.flatten(), self.centroid.flatten()) + i.radius for i in self.children])

    def compute_centroid(self):
        if self.leaf:
            if self.points == None or len(self.points) == 0:
                return None
            return np.mean([point for point in self.points],axis=0)
        else:
            return np.mean([i.centroid for i in self.children],axis=0)

    def intersects_point(self, point):
        return self.radius > distance.euclidean(point.flatten(), self.centroid.flatten())

    
    def update_bounding_envelope(self):
        self.centroid = self.compute_centroid()
        self.radius = self.compute_radius()

    def find_closest_child(self, target):
        return min(self.children, key=lambda x: distance.euclidean(x.centroid.flatten(), target.flatten()) if x.centroid is not None else float('inf'))

    def split(self, m):
        if self.leaf:
            max_variance_index = self.direction_of_max_variance()
            values = np.array([point[max_variance_index] for point in self.points])
            values.sort()

            best_split_index = None
            best_variance_reduction = float('inf')

            for i in range(m, len(values) - m + 1):
                values_left = values[:i]
                values_right = values[i:]

                variance_reduction = self.min_variance_split(values_left, values_right)

                if variance_reduction < best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_split_index = i
            points_left = self.points[:best_split_index]
            points_right = self.points[best_split_index:]
            newNode1 = SSnode(points=points_left, leaf=True)
            newNode2 = SSnode(points=points_right, leaf=True)
            data = []
            for i in range(len(newNode1.points)):
                if np.array_equal(self.points[i], newNode1.points[i]):
                    data.append(self.data[i])
            data = []
            for i in range(len(newNode1.points)):
                if np.array_equal(self.points[i], newNode2.points[i]):
                    data.append(self.data[i])
            newNode1.data = data    
            newNode2.data = data
            return newNode1, newNode2
        else:
            max_variance_index = self.direction_of_max_variance()
            values = np.array([child.centroid[max_variance_index] for child in self.children])
            values.sort()
            
            best_split_index = None
            best_variance_reduction = float('inf')

            for i in range(m, len(values) - m + 1):
                values_left = values[:i]
                values_right = values[i:]

                variance_reduction = self.min_variance_split(values_left, values_right)

                if variance_reduction < best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_split_index = i

            children_left = self.children[:best_split_index]
            children_right = self.children[best_split_index:]

            newNode1 = SSnode(children=children_left)
            newNode2 = SSnode(children=children_right)

            return newNode1, newNode2

    def min_variance_split(self, values_left, values_right):
        left_variance = np.var(values_left)
        right_variance = np.var(values_right)

        total_variance = (len(values_left) * left_variance + len(values_right) * right_variance) / (len(values_left) + len(values_right))

        return total_variance

    def direction_of_max_variance(self):
        if self.leaf:
            variances = np.var(self.points, axis=0)
        else:
            variances = np.var([i.centroid for i in self.children] , axis=0)
        max_variance_index = np.argmax(variances)
        return max_variance_index

    def get_entries_centroids(self):
        return [np.mean(c, axis=0) for c in self.points]
    
    def printNode(self, indent=0):
        print('\t' * indent, f'Centroid: {self.centroid}, Radius: {self.radius}')
        for child in self.children:
            child.printNode(indent + 2)



class SSTree:
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
                print(f"'{filename}' does not exist.")
                self.M = None
                self.m = None
                self.root = None


    def insert(self, point, data=None):
        if self.root is None:
            self.root = SSnode(leaf=True, points=[point], data=[data])
        else:
            node = self._choose_leaf(self.root, point)
            if tuple(point) in [tuple(p) for p in node.points]:
                return
            node.points.append(point)
            node.data.append(data)
            node.update_bounding_envelope()
            self.update_node(node)

        return self.root
    

    def update_node(self,node):
        if (node.leaf and len(node.points) > self.M) or (node.leaf == False and len(node.children) > self.M):
            if node.parent is None:
                new_node1, new_node2 = node.split(self.m)
                self.root = SSnode(children=[new_node1, new_node2])
                new_node1.parent = self.root
                new_node2.parent = self.root
            else:
                new_node1, new_node2 = node.split(self.m)
                node.parent.children.remove(node)
                node.parent.children.extend([new_node1, new_node2])
                new_node1.parent = node.parent
                new_node2.parent = node.parent
                node.parent.update_bounding_envelope()
                self.update_node(node.parent)


    def _choose_leaf(self, node, point):
        if node.leaf:
            return node
        while node.leaf is False:
            node = node.find_closest_child(point)
    
        return self._choose_leaf(node, point)

    def search(self, target):
        return self._search(self.root, target)

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

    def knn(self, q, k=3):
        L = PriorityQueue()
        Dk = [float('inf')]

        self.depth_first_search(q, k, self.root, L, Dk)
        nearest_neighbors = []

        while not L.empty():
            _, data = L.get()
            point = data['point']
            path = data['path']
            nearest_neighbors.append({'point': point, 'path': path})

        return nearest_neighbors[::-1]

    def depth_first_search(self, q, k, node, L, Dk):
        if node.leaf:
            for i, point in enumerate(node.points):
                dist = distance.euclidean(point, q)
                if dist < Dk[0]:
                    if L.qsize() == k:
                        L.get()
                    L.put((-dist, {'point': point, 'path': node.data[i]}))
                    if L.qsize() == k:
                        Dk[0] = -L.queue[0][0]
        else:
            for child in node.children:
                if child.intersects_point(q):
                    self.depth_first_search(q, k, child, L, Dk)
                    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
            
    def print(self):
        if self.root is not None:
            self.root.printNode()
        else:
            print("El árbol está vacío.")
